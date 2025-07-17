
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

def transcribe_audio(input_file, output_file, model_size="base"):
    """
    Transcribe audio/video file using OpenAI Whisper
    
    Args:
        input_file (str): Path to input audio/video file (.m4a or .mp4)
        output_file (str): Path to output transcript text file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
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
    
    print(f"Transcribing: {input_file}")
    result = model.transcribe(input_file)
    
    # Extract transcript text
    transcript = result["text"]
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    print(f"Transcript saved to: {output_file}")
    return transcript

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files using OpenAI Whisper")
    parser.add_argument("input_file", help="Input audio/video file (.m4a or .mp4)")
    parser.add_argument("output_file", help="Output transcript text file")
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: base)")
    
    args = parser.parse_args()
    
    try:
        transcribe_audio(args.input_file, args.output_file, args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()