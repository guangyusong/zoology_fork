import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs

def plot(
    df: pd.DataFrame,
    max_seq_len: int = 512,
    data_key="data.input_seq_len",
    model_key="model.sequence_mixer.1.name",
    x_lab="Sequence Length",
):
    
    plot_df = df.groupby([
        model_key,
        "model.d_model",
        data_key,
    ])["valid/accuracy"].max().reset_index()

    run_dir = "./var/sim_data/code/petting-zoo/"

    # remove nan 'valid/accuracy' 
    plot_df = plot_df[~plot_df["valid/accuracy"].isna()] 
    
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df[plot_df[data_key] <= max_seq_len],
        y="valid/accuracy",
        col=data_key,
        x="model.d_model",
        hue=model_key,
        kind="line",
        marker="o",
        height=2.25,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="Model dimension")

    # Set custom x-ticks
    ticks = [64, 128, 256, 512] # Modify this list as needed
    for ax in g.axes.flat:
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # This will keep the tick labels as integers rather than in scientific notation

    # Set custom y-ticks
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    for ax in g.axes.flat:
        ax.set_yticks(y_ticks)

    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f"{x_lab}: {title}")


if __name__ == "__main__" :
    experiment = 'num_kvs'

    launch_ids = [
        # "default-2023-12-10-09-20-16",
        "default-2023-12-10-09-20-48",
        # "default-2023-12-10-08-47-05",
        # "default-2023-12-10-08-39-31",
        # "default-2023-12-11-17-34-25",
        "default-2024-01-26-02-11-56"
    ]
    
    
    df = fetch_wandb_runs(
        launch_id=launch_ids, 
        project_name="zoology"
    )

    model_key1 = "model.sequence_mixer.name"
    model_key2 = "model.sequence_mixer.1.name"

    data_key1 = "data.builder.kwargs.num_kv_pairs"
    data_key2 = "data.0.builder.kwargs.num_kv_pairs"
    
    # set keys
    data_key = data_key2
    model_key = model_key1
    df[f"{model_key1}"] = df[f"{model_key1}"].fillna(df[f"{model_key2}"])
    
    # set plotting info
    x_key = data_key1
    x_lab = "Num. KV Pairs"
    df[f"{data_key1}"] = df[f"{data_key1}"].fillna(df[f"{data_key2}"])


    plot(df=df, max_seq_len=1024, data_key=x_key, model_key=model_key, x_lab=x_lab)
    print(f"Length of DF = {len(df)}")
    output_file = f"results_{experiment}.png"
    print(f"{output_file}")
    plt.savefig(output_file)
