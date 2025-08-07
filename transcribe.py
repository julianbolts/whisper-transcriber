"""
Audio/Video transcription using OpenAI Whisper
Supports .m4a and .mp4 files

Usage:
python transcribe.py audio.m4a transcript.txt
python transcribe.py video.mp4 output.txt --model medium
"""

import argparse
import whisper
import os
import sys

def transcribe_audio(input_file, output_file, model_size="base", snippet_size=5):
    """
    Transcribe audio/video file using OpenAI Whisper with configurable snippet timestamps
    
    Args:
        input_file (str): Path to input audio/video file (.m4a or .mp4)
        output_file (str): Path to output transcript text file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
        snippet_size (int): Size of each snippet in seconds (default: 5)
    """
    
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Validate file extension
    valid_extensions = ['.m4a', '.mp4']
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in valid_extensions:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: {valid_extensions}")
    
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing with word-level timestamps: {input_file}")
    result = model.transcribe(input_file, word_timestamps=True)
    
    # Group words by snippet intervals
    timestamped_transcript = create_snippet_intervals(result, snippet_size)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(timestamped_transcript)
    
    print(f"Timestamped transcript saved to: {output_file}")
    return timestamped_transcript

def create_snippet_intervals(result, snippet_size):
    """
    Create transcript with configurable snippet intervals from Whisper word timestamps
    
    Args:
        result: Whisper transcription result with word timestamps
        snippet_size (int): Size of each snippet in seconds
        
    Returns:
        str: Formatted transcript with [MM:SS - MM:SS] timestamps
    """
    lines = []
    
    # Extract all words with timestamps from all segments
    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            all_words.extend(segment["words"])
    
    if not all_words:
        # Fallback to segment-level timestamps if word-level not available
        for segment in result["segments"]:
            start_time = int(segment["start"])
            end_time = int(segment["end"])
            start_minutes = start_time // 60
            start_seconds = start_time % 60
            end_minutes = end_time // 60
            end_seconds = end_time % 60
            timestamp = f"[{start_minutes}:{start_seconds:02d} - {end_minutes}:{end_seconds:02d}]"
            lines.append(f"{timestamp} {segment['text'].strip()}")
        return "\n".join(lines)
    
    # Group words by snippet intervals
    current_snippet_start = 0
    current_words = []
    
    for word in all_words:
        word_second = int(word["start"])
        
        # If we've moved to a new snippet interval, output the previous snippet
        while current_snippet_start + snippet_size <= word_second:
            if current_words:
                start_minutes = current_snippet_start // 60
                start_seconds = current_snippet_start % 60
                end_minutes = (current_snippet_start + snippet_size - 1) // 60
                end_seconds = (current_snippet_start + snippet_size - 1) % 60
                timestamp = f"[{start_minutes}:{start_seconds:02d} - {end_minutes}:{end_seconds:02d}]"
                text = " ".join(current_words).strip()
                if text:  # Only add non-empty lines
                    lines.append(f"{timestamp} {text}")
                current_words = []
            current_snippet_start += snippet_size
        
        # Add word to current snippet if it falls within the current interval
        if current_snippet_start <= word_second < current_snippet_start + snippet_size:
            current_words.append(word["word"].strip())
    
    # Handle any remaining words
    if current_words:
        start_minutes = current_snippet_start // 60
        start_seconds = current_snippet_start % 60
        end_minutes = (current_snippet_start + snippet_size - 1) // 60
        end_seconds = (current_snippet_start + snippet_size - 1) % 60
        timestamp = f"[{start_minutes}:{start_seconds:02d} - {end_minutes}:{end_seconds:02d}]"
        text = " ".join(current_words).strip()
        if text:
            lines.append(f"{timestamp} {text}")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using OpenAI Whisper")
    parser.add_argument("input_file", help="Input audio/video file (.m4a or .mp4)")
    parser.add_argument("output_file", help="Output transcript text file")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--snippet-size", type=int, default=5,
                       help="Size of each snippet in seconds (default: 5)")
    
    args = parser.parse_args()
    
    try:
        transcribe_audio(args.input_file, args.output_file, args.model, args.snippet_size)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()