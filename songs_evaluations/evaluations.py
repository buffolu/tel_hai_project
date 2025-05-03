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


def compare(original_sources_path, estimated_sources_path):
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

            results["SDR"].append({name: sdr_val[0]})
            results["SIR"].append({name: sir_val[0]})
            results["SAR"].append({name: sar_val[0]})
        return results
    except FileNotFoundError:
        print(f"I couldn't find the music files of the given song. please ensure you have the song directory with the correct name and data as the instructions")
    except Exception as e:
        print(f"An error occurred while evaluating the song: {e}")


def song_evaluation(evaluation_path, original_sources_path, prior_attack_sources_path, after_attack_sources_path, after_defence_sources_path):
    evaluation_scores = {
        "demucs_default": compare(original_sources_path,prior_attack_sources_path),
        "attack_effect_evaluation":  compare(prior_attack_sources_path, after_attack_sources_path),
        "defence_effect_evaluation": compare(original_sources_path,after_defence_sources_path)
    }
    results_saving_path = os.path.join(evaluation_path, "evaluations_scores.json")
    os.makedirs(os.path.dirname(results_saving_path), exist_ok=True)

    with open(results_saving_path, "w") as f:
        json.dump(convert_np_floats_to_python(evaluation_scores), f, indent=4, allow_nan=True)

example="../originals/AM_Contra_-_Heart_Peripheral"
song_evaluation(example, example, example,example,example)