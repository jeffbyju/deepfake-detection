#!/bin/bash

# Define directory and log file
OUTPUT_DIR="runs/20epochs-bi_lstm-fixed_focal-30seq"
LOG_FILE="${OUTPUT_DIR}/log.txt"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script and log output
python3 -u demo.py --output_dir="$OUTPUT_DIR" --bi_lstm_bool=true --attention_bool=false --seq_length=40 --csv_file="./final_celeb_df2" --base_dir="CELEB-DF-2" 2>&1 | tee "$LOG_FILE"