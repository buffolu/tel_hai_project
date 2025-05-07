import os
from demucs_separate import demucs_separate

if __name__ == "__main__":
    try:
        songs_folder = "songs"
        for song_folder_name in os.listdir(songs_folder):
            song_folder_path = os.path.join(songs_folder, song_folder_name)
            if not os.path.isdir(song_folder_path):
                continue
            original_song_path = os.path.join(song_folder_path, "original_song")
            #if there is no folder name "original_song" raise an error
            if not os.path.isdir(original_song_path):
                raise FileNotFoundError(f"'original_song' folder not found in {song_folder_path}")
            if not os.path.isdir(song_folder_path):
                raise FileNotFoundError(f"{song_folder_name} is not a directory")
            song_stem_path=os.path.join(original_song_path,"Al James - Schoolboy Facination.stem.mp4")
            # if there is no file name song_stem_path raise an error
            if not os.path.isfile(song_stem_path):
                raise FileNotFoundError(f"original song not found: {song_stem_path}")

            # if there is a trimmed_song folder, delete it
            trimmed_song_dir=os.path.join(song_folder_path,"trimmed_song")
            if not os.path.isdir(trimmed_song_dir):
                raise FileNotFoundError(f"you should trim the song before running the attacking script: {song_folder_path}")

            #create the necessary folders
            separation_prior_attack_path=os.path.join(song_folder_path, "separation_prior_attack")
            os.makedirs(separation_prior_attack_path, exist_ok=True)
            os.makedirs(os.path.join(song_folder_path, "attacked_song"), exist_ok=True)
            os.makedirs(os.path.join(song_folder_path, "separation_after_attack"), exist_ok=True)
            os.makedirs(os.path.join(song_folder_path, "defended_song"), exist_ok=True)
            os.makedirs(os.path.join(song_folder_path, "separation_after_attack_and_defence"), exist_ok=True)
            os.makedirs(os.path.join(song_folder_path, "evaluations"), exist_ok=True)


            trimmed_song_path=os.path.join(trimmed_song_dir,f"mixture.wav")
            # let demucs estimate seperation on the original song
            demucs_separate(trimmed_song_path,separation_prior_attack_path )
            # run attacking script
            # run defending script
            # run evaluation script

    except Exception as e:
        print("An error occurred:", e)
