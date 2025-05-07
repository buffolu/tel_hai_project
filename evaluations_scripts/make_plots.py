import numpy as np
import matplotlib.pyplot as plt
import os
import json


def song_metric_plot(evaluation_scores, evaluation_metric,saving_path):
    """
    Plots a grouped bar chart for a single song, showing SDR or SIR per channel and comparison type,
    excluding the 'mixture' channel.

    :param evaluation_scores: dict with keys 'demucs_default', 'attack_effect_evaluation', 'defence_effect_evaluation',
                              each mapping to a dict of {channel: {metric: value}}.
    :param evaluation_metric: str, either "SDR" or "SIR"
    """
    print(evaluation_scores)
    channels = ['bass', 'drums', 'vocals', 'other']
    comparisons = ['demucs_default', 'attack_effect_evaluation', 'defence_effect_evaluation']
    labels = ['Demucs', 'Attack Effect', 'Defence Effect']
    colors = ['skyblue', 'salmon', 'lightgreen']

    x = np.arange(len(channels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (key, label, color) in enumerate(zip(comparisons, labels, colors)):
        values = [
            next((d[ch] for d in evaluation_scores[key].get(evaluation_metric, []) if ch in d), 0)
            for ch in channels
        ]
        ax.bar(x + i * width - width, values, width, label=label, color=color)
        for j, value in enumerate(values):
            ax.text(x[j] + i * width - width, value + 0.3, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(evaluation_metric)
    ax.set_title(f'{evaluation_metric} per Channel')
    ax.set_xticks(x)
    ax.set_xticklabels([ch.capitalize() for ch in channels])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"./{saving_path}/{evaluation_metric}_Evaluation.png")


def plot_per_song(saving_path, evaluations_json_path):
    if not os.path.exists(evaluations_json_path):
        raise FileNotFoundError(f"No such file: {evaluations_json_path}")
    with open(evaluations_json_path, 'r') as f:
        eval_json = json.load(f)

    os.makedirs(saving_path, exist_ok=True)
    song_metric_plot(eval_json, "SDR", saving_path)
    song_metric_plot(eval_json, "SIR", saving_path)
    song_metric_plot(eval_json, "SAR", saving_path)



example="../originals/AM_Contra_-_Heart_Peripheral"
jsont="../originals/AM_Contra_-_Heart_Peripheral/evaluations_scores.json"
plot_per_song(example, jsont)