import yaml
import itertools
from main import run

param_grid = {
    "batch_size": [16, 32, 64, 128, 256],
    "lr": [0.12, 0.09, 0.06, 0.03, 0.01],
    "model_name": ["test"],
    "class_name": ["test"],
    "epochs": [2],
}

# 조합 생성
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
