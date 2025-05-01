from mir_eval import separation
import os
import librosa
import numpy as np
import json

def convert_np_floats_to_python(obj):
    """
    Recursively converts numpy float32/float64 types to native Python float
    to allow JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_np_floats_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_floats_to_python(elem) for elem in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    else:
        return obj

def get_seperated_channels(folder_path: str) -> dict[str, np.ndarray]:
    separated_files_path = {
        "vocals": f'{folder_path}/vocals.wav',
        "drums": f'{folder_path}/drums.wav',
        "bass": f'{folder_path}/bass.wav',
        "other": f'{folder_path}/other.wav'
    }
    sources_files: dict[str, np.ndarray] = {}

    for name,path in separated_files_path.items():
        if not os.path.exists(path):
            raise FileNotFoundError
        audio, samplerate = librosa.load(path, sr=None, mono=True)
        sources_files[name] = audio
    return sources_files


def song_evaluation(song_dir_name, original_sources_path, estimated_sources_path):
    try:
        if not os.path.exists(original_sources_path) or not os.path.exists(estimated_sources_path):
            raise FileNotFoundError

        estimated_sources=get_seperated_channels(estimated_sources_path)
        original_sources=get_seperated_channels(original_sources_path)

        results = {
            "SDR": [],
            "SIR": [],
            "SAR": [],
        }

        for idx, name in enumerate(["vocals", "drums", "bass", "other"]):
            sdr_val, sir_val, sar_val, perm = separation.bss_eval_sources(original_sources[name], estimated_sources[name])

            results["SDR"].append({name: sdr_val[idx]})
            results["SIR"].append({name: sir_val[idx]})
            results["SAR"].append({name: sar_val[idx]})

        results_saving_path=os.path.join("originals",song_dir_name,"evaluations_scores.json")
        with open(results_saving_path, "w") as f:
            json.dump(convert_np_floats_to_python(results), f, indent=4, allow_nan=True)

        return results
    except FileNotFoundError:
        print(f"I couldn't find the music files of the given song. please ensure you have the song directory with the correct name and data as the instructions")
    except Exception as e:
        print(f"An error occurred while evaluating the song: {e}")

song_evaluation("AM_Contra_-_Heart_Peripheral", "../originals/AM_Contra_-_Heart_Peripheral", "../separation_after_attack_and_deface/AM Contra - Heart Peripheral.stem")