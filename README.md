# whisper-transcriber

util script to transcribe an audio or video file using python and openai whisper

## usage

Always enable the venv:

`source  ./.venv/bin/activate`

the first time, install things with

`pip install openai-whisper`

and how to run the script with all params present

`python transcribe.py input_file.m4a output_file.txt --model medium --snippet-size 10`

real example:

`python transcribe.py input/GMT20250715-162056_Recording.m4a output/csa-onboarding-shadowing.transcript.txt`

required options:
- input_file path
- output_file path

default values for non-required options:
- model = 'base'    # models: tiny, base, small, medium, large
- snippet-size = 5  # the duration for each line of transcribed audio in seconds
