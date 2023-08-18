#!/bin/bash

# Set the input and output directories
input_directory="./dl"
output_directory="./dl"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through all wav files in the input directory
for wav_file in "$input_directory"/*.wav; do
    if [ -f "$wav_file" ]; then
        # Get the filename without the directory path and extension
        filename=$(basename "$wav_file")
        filename_no_ext="${filename%.*}"

        # Output filename in the output directory
        output_filename="$output_directory/${filename_no_ext}_reencoded.wav"

        # Reencode the wav file using ffmpeg
        ffmpeg -i "$wav_file" -acodec pcm_s16le -ar 44100 "$output_filename"

        echo "Reencoded: $filename -> ${filename_no_ext}_reencoded.wav"
    fi
done

echo "Reencoding process completed."
