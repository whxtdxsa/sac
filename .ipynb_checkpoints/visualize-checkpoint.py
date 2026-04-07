import os
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

folder_name = "ex_pendulum"
file_name = "metrics.csv"
model_dirs = {
    "basic_reward": f"{folder_name}/al0.0_crCriterionByRewardScaling_lr0.0003_hi256/{file_name}",
    "scaled_reward": f"{folder_name}/al0.0_crCriterion_lr0.0003_hi256/{file_name}",
    # "alpha0.0": f"{folder_name}/al0.0_lr0.0003_hi256/{file_name}",
    # "alpha0.2": f"{folder_name}/al0.2_lr0.0003_hi256/{file_name}",
    # "alpha0.5": f"{folder_name}/al0.5_lr0.0003_hi256/{file_name}",
    # "alpha0.8": f"{folder_name}/al0.8_lr0.0003_hi256/{file_name}",
}

colors_25 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",  # Tableau 10
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",  # D3 Category20b
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",  # D3 Category20c
    "#f781bf",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#cab2d6",  # Pastel from Set3
]

plt.figure(figsize=(10, 6))

for i, (model_name, path) in enumerate(model_dirs.items()):
    df = pd.read_csv(path)
    color = colors_25[i]

    # plt.plot(df["epoch"], df["train_loss"], label=f"{model_name}_train", color=color, linewidth=1.0)
    plt.plot(
        df["epsiode"], df["reward"], label=f"{model_name}", color=color, linewidth=3.0
    )

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Scaled vs Basic Reward with alpha=0.0")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize="small")
plt.tight_layout()
plt.grid(True)
# plt.ylim(0.08, 0.2)
plt.savefig("scaled_reward.png")
