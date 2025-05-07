#!/bin/bash

# Set variables
INPUT_DIR="songs"
START=0         # Start time in seconds
DURATION=5     # Duration in seconds

python3 trim_song.py \
    "Al James - Schoolboy Facination" \
    --start "$START" \
    --duration "$DURATION"
