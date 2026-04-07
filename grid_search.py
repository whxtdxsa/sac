import yaml
import itertools
from main import run

param_grid = {
    "alpha": [0.0, 0.2, 0.5, 0.8],
    # "hidden_size": [128, 256, 512],
    # "lr": [1.0e-4, 3.0e-4],
    # "criterion_name": ["CriterionByRewardScaling", "Criterion"],
}

# Make combinations
combinations = list(itertools.product(*param_grid.values()))

with open("config/default.yaml") as f:
    base_cfg = yaml.safe_load(f)

for i, values in enumerate(combinations):
    cfg = base_cfg.copy()
    for k, v in zip(param_grid.keys(), values):
        cfg[k] = v

    print(f"\n[Experiment {i + 1}/{len(combinations)}]")
    print(cfg)
    run(cfg)
