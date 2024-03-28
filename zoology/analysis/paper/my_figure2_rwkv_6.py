import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs

def plot(df: pd.DataFrame, max_seq_len: int = 512):
    
    # Rename models
    df['model_name'] = df['model.sequence_mixer.name'].replace({
        'zoology.mixers.rwkv.RWKVTimeMixer': 'RWKV-4',
        'zoology.mixers.rwkv5.RWKV_TimeMix_RWKV5': 'Eagle (RWKV-5)',
        'zoology.mixers.rwkv6.RWKV_Tmix_x060': 'RWKV-6'
    })
    
    plot_df = df.groupby([
        "model_name",
        "model.d_model",
        "data.input_seq_len",
    ])["valid/accuracy"].max().reset_index()

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df[plot_df["data.input_seq_len"] <= max_seq_len],
        y="valid/accuracy",
        col="data.input_seq_len",
        x="model.d_model",
        hue="model_name",
        kind="line",
        marker="o",
        height=2.25,
        aspect=1.5,
        facet_kws={'sharex': False, 'sharey': True}  # Ensure x-axis are not scaled proportionally
    )
    g.set(ylabel="Accuracy", xlabel="Model dimension")

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
        ax.set_title(f"Sequence Length: {title}")

    # Remove the legend title (previously 'model.sequence_mixer.name')
    g._legend.set_title('')

if __name__ == "__main__":
    df = fetch_wandb_runs(
        launch_id=[
            "default-2024-02-24-23-11-25"
            ],
        project_name="zoology-rwkv-6"
    )

    print(df.columns)
    plot(df=df, max_seq_len=1024)
    plt.savefig("rwkv-6.pdf")

# import os
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# from zoology.analysis.utils import fetch_wandb_runs

# def plot(df: pd.DataFrame, max_seq_len: int = 512):
    
#     # Rename models
#     df['model_name'] = df['model.sequence_mixer.name'].replace({
#         'zoology.mixers.rwkv.RWKVTimeMixer': 'RWKV 4',
#         'zoology.mixers.rwkv5.RWKV_TimeMix_RWKV5': 'RWKV 5.2'
#     })
    
#     plot_df = df.groupby([
#         "model_name",
#         "model.d_model",
#         "data.input_seq_len",
#     ])["valid/accuracy"].max().reset_index()

#     # Convert 'model.d_model' to string for categorical plotting
#     plot_df['model.d_model'] = plot_df['model.d_model'].astype(str)

#     sns.set_theme(style="whitegrid")
#     g = sns.relplot(
#         data=plot_df[plot_df["data.input_seq_len"] <= max_seq_len],
#         y="valid/accuracy",
#         col="data.input_seq_len",
#         x="model.d_model",
#         hue="model_name",
#         kind="line",
#         marker="o",
#         height=2.25,
#         aspect=1.5,
#     )
#     g.set(ylabel="Accuracy", xlabel="Model dimension")

#     # Set custom x-ticks
#     ticks = ['64', '128', '256', '512']  # Ensure these are strings
#     for ax in g.axes.flat:
#         ax.set_xticks(ticks)

#     # Set custom y-ticks
#     y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
#     for ax in g.axes.flat:
#         ax.set_yticks(y_ticks)

#     for ax, title in zip(g.axes.flat, g.col_names):
#         ax.set_title(f"Sequence Length: {title}")

#     # Remove the legend title
#     g._legend.set_title('')

# if __name__ == "__main__":
#     df = fetch_wandb_runs(
#         launch_id=[
#             "default-2024-01-14-15-46-47", # RWKV 64 & 128
#             "default-2024-01-15-11-49-40", # RWKV 128
#             "default-2024-01-15-16-17-41", # RWKV 256
#             "default-2024-01-16-15-37-55" # RWKV 512
#             ],
#         project_name="zoology-rwkv"
#     )

#     print(df.columns)
#     plot(df=df, max_seq_len=1024)
#     plt.savefig("rwkv4 vs 5.pdf")
