# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:12:32 2025

@author: igor
"""

import sys
import yt_dlp
import os

# Check if a URL and a filename were provided as arguments
if len(sys.argv) < 2:
    print("Please provide a YouTube URL as an argument.")
    sys.exit(1)

# Get the YouTube URL from the command line argument
song_url = sys.argv[1]

# If a custom file name is provided, use it; otherwise, use the video title
if len(sys.argv) > 2:
    custom_filename = sys.argv[2]
else:
    custom_filename = None

# Ensure the 'songs' folder exists
if not os.path.exists('songs'):
    os.makedirs('songs')

# Set up download options
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'aac',
        'preferredquality': '192',
    }],
    'outtmpl': 'songs/%(title)s.%(ext)s',  # Default to video title if no custom filename is given
}

# If a custom filename was provided, modify the 'outtmpl' to use it
if custom_filename:
    ydl_opts['outtmpl'] = f'songs/{custom_filename}.%(ext)s'

# Download the song
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([song_url])


