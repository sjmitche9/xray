# asha_tune_optuna.py
import os, sys, yaml

from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Strongly recommended for tuning (cheap, stable).
METRIC_NAME = "val_score"
METRIC_MODE = "max"


def trial_dirname_creator(trial):
    # Short, deterministic, Windows-safe
    return f"t{trial.trial_id[:8]}"


def _write_overrides_to_yaml(base_cfg_path: str, out_cfg_path: str, overrides: dict) -> None:
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tr = cfg.setdefault("TRAINING", {})
    sch = tr.setdefault("LR_SCHEDULER", {})

    # ---- DEBUG SETTINGS ----
    tr["TRAIN_ON_THREE_BATCHES"] = False  # leave your debug switch intact

    # ---- FAST TUNE FLAGS ----
    # Disable the expensive validation diffusion preview paths.
    tr["FAST_TUNE"] = True
    tr["DISABLE_PREVIEW_RECON"] = True
    tr["DISABLE_PREVIEW_CFG"] = True
    tr["RUN_PREVIEWS_EVERY"] = 0   # <-- hard off, regardless of base config
    tr["SCORE_ONLY"] = True       # <--- new

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

    # budget knobs (you already have these)
    tr["EPOCHS"] = int(overrides["epochs"])
    tr["CHUNK_LIMIT"] = int(overrides["chunk_limit"])

    os.makedirs(os.path.dirname(out_cfg_path), exist_ok=True)
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def trainable(config: dict):
    import os
    import sys
    import io

    # ---- 1) Ensure sys.stdout/sys.stderr have .isatty() (Ray Tee sometimes doesn't)
    class _TtyProxy:
        def __init__(self, stream):
            self._stream = stream
        def isatty(self):
            return False
        def __getattr__(self, name):
            return getattr(self._stream, name)

    if sys.stdout is not None and not hasattr(sys.stdout, "isatty"):
        sys.stdout = _TtyProxy(sys.stdout)
    if sys.stderr is not None and not hasattr(sys.stderr, "isatty"):
        sys.stderr = _TtyProxy(sys.stderr)

    # ---- 2) Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
    # Preferred: Python 3.7+ reconfigure
    for sname in ("stdout", "stderr"):
        s = getattr(sys, sname)
        try:
            # Works for TextIOWrapper
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    # Fallback: if stream has a buffer, wrap it with UTF-8 TextIOWrapper
    for sname in ("stdout", "stderr"):
        s = getattr(sys, sname)
        try:
            if hasattr(s, "buffer") and isinstance(s.buffer, (io.BufferedIOBase, io.RawIOBase)):
                setattr(
                    sys,
                    sname,
                    io.TextIOWrapper(s.buffer, encoding="utf-8", errors="backslashreplace", line_buffering=True),
                )
        except Exception:
            pass

    # Final fallback: wrap write() so it never crashes on encoding
    class _SafeWrite:
        def __init__(self, stream):
            self._stream = stream
        def write(self, msg):
            try:
                return self._stream.write(msg)
            except UnicodeEncodeError:
                # Last resort: strip to ASCII-ish
                safe = msg.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")
                return self._stream.write(safe)
        def flush(self):
            try:
                return self._stream.flush()
            except Exception:
                return None
        def isatty(self):
            return False
        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _SafeWrite(sys.stdout)
    sys.stderr = _SafeWrite(sys.stderr)

    # ---- Your original logic
    os.chdir(PROJECT_ROOT)

    os.environ["WANDB_MODE"] = "online"
    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
    os.environ["XRAY_PROJECT_ROOT"] = PROJECT_ROOT

    trial_dir = session.get_trial_dir()

    base_cfg = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    trial_cfg = os.path.join(trial_dir, "config", "config.yaml")
    os.makedirs(os.path.dirname(trial_cfg), exist_ok=True)
    _write_overrides_to_yaml(base_cfg, trial_cfg, config)

    os.chdir(trial_dir)

    trial_id = session.get_trial_id()
    os.environ["WANDB_NAME"] = f"trial_{trial_id}"

    import lora_unet_transfer_train as train_mod
    train_mod.main()


if __name__ == "__main__":
    param_space = {
        "lr": tune.loguniform(5e-6, 5e-5),
        "min_lr": tune.loguniform(1e-6, 1e-5),
        "max_grad_norm": tune.uniform(0.5, 2.0),
        "lora_dropout": tune.uniform(0.0, 0.15),
        "cfg_dropout": tune.uniform(0.0, 0.10),
        "beta": tune.uniform(0.8, 1.2),
        "ssim_weight": tune.uniform(0.0, 0.12),
        "epochs": 35,
        "chunk_limit": 10,
}

    scheduler = ASHAScheduler(
        metric=METRIC_NAME,
        mode=METRIC_MODE,
        time_attr="epoch",
        max_t=int(param_space["epochs"]),
        grace_period=8,
        reduction_factor=3,
    )

    search_alg = OptunaSearch(
        metric=METRIC_NAME,
        mode=METRIC_MODE,
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 16, "gpu": 1}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=50,
            max_concurrent_trials=1,
            trial_dirname_creator=trial_dirname_creator,
        ),
        run_config=tune.RunConfig(
            name="lora_unet_optuna_fast",
            storage_path=os.path.abspath("./ray_results"),
            log_to_file=True,
        ),
    )

    tuner.fit()