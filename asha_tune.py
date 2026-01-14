# asha_tune_optuna.py
import os, sys, yaml

from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

METRIC_NAME = "val_recon_ssim_mean"


def trial_dirname_creator(trial):
    # Short, deterministic, Windows-safe
    return f"t{trial.trial_id[:8]}"


def _write_overrides_to_yaml(base_cfg_path: str, out_cfg_path: str, overrides: dict) -> None:
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    tr = cfg.setdefault("TRAINING", {})
    sch = tr.setdefault("LR_SCHEDULER", {})

    # ---- DEBUG SETTINGS ----
    # For fast local debug; set False for real runs
    tr["TRAIN_ON_THREE_BATCHES"] = False

    # fixed (dataset/choices)
    tr["TEXT_TOKENS"] = 16
    tr["LORA_R"] = 64
    tr["LORA_ALPHA"] = 64

    # tuned
    tr["LEARNING_RATE"] = float(overrides["lr"])
    tr["MAX_GRAD_NORM"] = float(overrides["max_grad_norm"])
    tr["CFG_DROPOUT"]   = float(overrides["cfg_dropout"])
    tr["SSIM_WEIGHT"]   = float(overrides["ssim_weight"])
    tr["BETA"]          = float(overrides["beta"])
    tr["LORA_DROPOUT"]  = float(overrides["lora_dropout"])
    sch["MIN_LR"]       = float(overrides["min_lr"])

    # budget knobs (fixed here)
    tr["EPOCHS"] = int(overrides["epochs"])
    tr["CHUNK_LIMIT"] = int(overrides["chunk_limit"])

    os.makedirs(os.path.dirname(out_cfg_path), exist_ok=True)
    with open(out_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def trainable(config: dict):
    # Be explicit about where "relative paths" should resolve from
    os.chdir(PROJECT_ROOT)

    # Disable W&B during Tune debug (safe even if trainer calls wandb.init/log)
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")
    os.environ["WANDB_SILENT"] = os.environ.get("WANDB_SILENT", "true")
    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
    os.environ["XRAY_PROJECT_ROOT"] = PROJECT_ROOT

    trial_dir = session.get_trial_dir()

    base_cfg = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    trial_cfg = os.path.join(trial_dir, "config", "config.yaml")
    os.makedirs(os.path.dirname(trial_cfg), exist_ok=True)
    _write_overrides_to_yaml(base_cfg, trial_cfg, config)

    # Make trainer read trial-local config/config.yaml
    os.chdir(trial_dir)

    import lora_unet_transfer_train as train_mod
    train_mod.main()


if __name__ == "__main__":
    # Optuna search space (continuous where it matters)
    # NOTE: tune.loguniform(low, high) requires 0 < low < high
    param_space = {
        "lr": tune.loguniform(5e-6, 8e-5),
        "min_lr": tune.loguniform(1e-6, 2e-5),
        "max_grad_norm": tune.uniform(0.3, 2.5),
        "lora_dropout": tune.uniform(0.0, 0.2),
        "cfg_dropout": tune.uniform(0.0, 0.15),
        # beta is SmoothL1 beta; loguniform is fine as long as >0
        "beta": tune.loguniform(0.5, 2.0),
        "ssim_weight": tune.uniform(0.0, 0.25),

        # budget knobs fixed for debug
        "epochs": 100,
        "chunk_limit": 36,
    }

    # With epochs=2, ASHA isn't very meaningful but is fine for wiring debug.
    # Keep grace_period <= max_t.
    scheduler = ASHAScheduler(
        metric=METRIC_NAME,
        mode="max",
        max_t=int(param_space["epochs"]),
        grace_period=10,
        reduction_factor=4,
    )

    search_alg = OptunaSearch(
        metric=METRIC_NAME,
        mode="max",
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 16, "gpu": 1}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,     # <-- this is the missing piece
            scheduler=scheduler,
            num_samples=50,             # <-- >1 so you actually test Optuna suggestions
            trial_dirname_creator=trial_dirname_creator
        ),
        run_config=tune.RunConfig(
            name="lora_unet_optuna_debug",
            storage_path=os.path.abspath("./ray_results"),
            log_to_file=True,
        ),
    )

    tuner.fit()