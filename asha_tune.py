# asha_tune.py
import os, sys, yaml, shutil
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler

# Adjust if your repo root is different
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Where your script lives (importable)
sys.path.insert(0, PROJECT_ROOT)

def _write_overrides_to_yaml(base_cfg_path, out_cfg_path, overrides: dict):
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- apply overrides (edit these few keys as you sweep) ----
    tr = cfg.setdefault("TRAINING", {})
    tr["LEARNING_RATE"] = overrides["lr"]
    tr["TEXT_TOKENS"]   = overrides["text_tokens"]
    tr["LORA_R"]        = overrides["lora_r"]
    tr["LORA_ALPHA"]    = overrides["lora_alpha"]
    # optionally shorten epochs for ASHA; leave your script untouched
    tr["EPOCHS"]        = overrides["epochs"]

    os.makedirs(os.path.dirname(out_cfg_path), exist_ok=True)
    with open(out_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def trainable(config):
    # Each trial runs in its own working dir; ensure imports find your code
    os.chdir(PROJECT_ROOT)

    # Put a trial-local config in ./config/config.yaml (no CLI needed)
    base_cfg = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    trial_cfg_dir = os.path.join(tune.get_trial_dir(), "config")
    trial_cfg = os.path.join(trial_cfg_dir, "config.yaml")

    # Copy any extra files your script expects next to config (optional)
    if not os.path.exists(trial_cfg_dir):
        os.makedirs(trial_cfg_dir, exist_ok=True)

    _write_overrides_to_yaml(base_cfg, trial_cfg, config)

    # Make your code read the trial-local config by changing CWD
    # so `with open("config/config.yaml")` resolves to trial copy.
    os.chdir(tune.get_trial_dir())

    # Import and run your training (no CLI). It will call tune.report() each epoch.
    import lora_unet_transfer_train_old as train_mod
    train_mod.main()

# ---------------- Search Space ----------------
search_space = {
    "lr": tune.grid_search([2e-5, 5e-5, 1e-4]),
    "text_tokens": tune.grid_search([4, 8]),
    "lora_r": tune.grid_search([8, 16]),
    "lora_alpha": tune.grid_search([8, 16, 32]),
    "epochs": 8,  # ASHA will early-stop many trials before this
}

scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=search_space["epochs"],
    grace_period=2,     # let each trial run at least 2 epochs
    reduction_factor=3, # promote top ~1/3
)

tuner = tune.Tuner(
    tune.with_resources(trainable, resources={"cpu": 4, "gpu": 1}),
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=scheduler),
    run_config=air.RunConfig(
        name="lora_unet_asha",
        local_dir=os.path.abspath("./ray_results"),
        log_to_file=True,   # capture stdout/stderr per trial
    ),
)

if __name__ == "__main__":
    tuner.fit()