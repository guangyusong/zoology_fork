import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs

def plot(df: pd.DataFrame, max_seq_len: int = 512):
    # Rename models
    df['model_name'] = df['model.sequence_mixer.name'].replace({
        'zoology.mixers.rwkv.RWKVTimeMixer': 'Raven (RWKV-4)',
        'zoology.mixers.rwkv5.RWKV_TimeMix_RWKV5': 'Eagle (RWKV-5)',
        'zoology.mixers.rwkv6.RWKV_Tmix_x060': 'Finch (RWKV-6)',
        'zoology.mixers.hyena.Hyena': 'Hyena',
        'zoology.mixers.mamba.Mamba': 'Mamba',
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
        x="model.d_model",
        col="data.input_seq_len",
        hue="model_name",
        hue_order=['Raven (RWKV-4)', 'Eagle (RWKV-5)', 'Finch (RWKV-6)', 'Hyena', 'Mamba'],
        kind="line",
        marker="o",
        height=2.5,
        aspect=1.1,
        facet_kws={'sharex': False, 'sharey': True},
    )
    
    for ax in g.axes.flat:
        for line in ax.get_lines():
            if line.get_label() == 'Finch (RWKV-6)':
                line.set_zorder(5)  # Higher z-order for RWKV-6
            else:
                line.set_zorder(1)  # Default z-order for other lines

    # Set labels and ticks
    g.set(ylabel="Accuracy", xlabel="Model dimension")
    ticks = [64, 128, 256, 512]
    for ax in g.axes.flat:
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    y_ticks = [0, 0.5, 0.75, 1.0]
    for ax in g.axes.flat:
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, 1.0)  # Adjust this range based on your data's distribution

    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f"Sequence Length: {title}")

    # Create a custom legend at the bottom
    g._legend.remove() 
    handles, labels = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

    # Adjust the bottom margin
    g.fig.subplots_adjust(bottom=0.12, top=0.95, hspace=0.4, wspace=0.05)

    # Save the plot
    g.fig.savefig("rwkv4v5v6.png", bbox_inches='tight')

if __name__ == "__main__":     
    df_v4_v5 = fetch_wandb_runs(
        launch_id=["default-2024-02-29-01-51-11"],
        project_name="zoology-rwkv-4-5"
    )
    
    df_v6 = fetch_wandb_runs(
        launch_id=["default-2024-02-24-23-11-25"],
        project_name="zoology-rwkv-6"
    )
    
    df_hyena = fetch_wandb_runs(
        launch_id=["default-2024-03-23-04-00-50"], 
        project_name="zoology-hyena"
    )
    
    df_mamba = fetch_wandb_runs(
        launch_id=["default-2024-03-10-12-48-24"], 
        project_name="zoology-mamba"
    )

    # Combine the dataframes
    df_combined = pd.concat([df_v4_v5, df_v6, df_hyena, df_mamba], ignore_index=True)

    print(df_combined.columns)
    plot(df=df_combined, max_seq_len=1024)
