import stempeg
import os

input_root = "./songs"
output_root = "./originals"
stem_names = ['mixture', 'drums', 'bass', 'other', 'vocals']

# go through all songs in the directory
for song in os.listdir(input_root):
    if song.endswith(".stem.mp4"):
        song_name = song.replace(".stem.mp4", "").replace(" ", "_")
        output_dir = os.path.join(output_root, song_name)

        # קריאה של הערוצים מתוך הקובץ
        audio, rate = stempeg.read_stems(song, stem_id=None)

        # יצירת תיקייה אם לא קיימת
        os.makedirs(output_dir, exist_ok=True)

        # כתיבת כל ערוץ לקובץ WAV
        for i, stem_audio in enumerate(audio):
            print(f"{stem_names[i]} shape: {stem_audio.shape}")
            output_path = os.path.join(output_dir, f"{stem_names[i]}.wav")
            stempeg.write_audio(output_path, stem_audio, rate)
