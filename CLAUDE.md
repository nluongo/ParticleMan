# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ParticleMan pre-trains transformer foundation models on particle-physics events (ATLAS ROOT/HDF5 data). Each event is a padded set of particles with features (pt, eta, phi, particle_id). The model is trained with masked-particle prediction ("pretrain") and can be fine-tuned for event classification ("classify"). Training runs on HPC (PBS, LCRC paths under `/lcrc/group/ATLAS/...`) with torchrun/DDP.

## Environment & commands

- Conda env: `conda env create -f environment.yml` (CPU) or `environment-gpu.yml`; then `pip install -e ".[dev]"` (`[logging]` adds mlflow).
- Tests live in top-level `tests/` (pytest `testpaths`). Run `pytest`, a single test with `pytest tests/test_particle_transformer.py::TestParticleConfig::test_default_config`.
- Lint/format/type: `ruff check src/`, `black src/`, `isort src/`, `flake8 src/`, `mypy src/` (line length 88, black profile). mypy is strict (`disallow_untyped_defs`).
- CI (`.github/workflows/ci.yml`) runs ruff, flake8, isort, black, mypy, pytest and bandit on pushes/PRs to `master`, `main` and `develop`.

## Running training

All entry points use Hydra with `configs/` as the config dir. They are run as scripts, not modules. They prepend `src/` to `sys.path`.

- `python scripts/train.py [overrides]`: the default root config is `configs/config.yaml`. Pick a different root with `--config-name bbllv08` (or `mc20_ttbar`). Override with `group=name` or `key.path=value`, e.g. `data=bbllv08_classify training=quick_test training.mode=classify model.d_model=256`. `--cfg job` prints the resolved config.
- Multi-GPU / multi-node, interactive (inside `qsub -I` / `salloc`, or on a GPU workstation): `bash scripts/jobs/launch.sh scripts/train.py <overrides>`. It detects the scheduler, nodes and GPUs. `LAUNCHER=torchrun` (the default), `mpiexec` (PBS) or `srun` (Slurm) picks the launch mode. Plain `torchrun --standalone --nproc_per_node=N scripts/train.py ...` also works on a single node.
- Batch: `python scripts/submit.py [--dry-run] [--config-name X] cluster=<site> resources=<size> [cluster.*=/resources.*=/job.*=...] <training overrides>`. It composes `configs/submit_config.yaml` (site settings in `configs/cluster/`, job size in `configs/resources/`), validates the training overrides by composing the training config, and runs `qsub`/`sbatch` on `scripts/jobs/launch.sh`. Settings reach the job as env vars, passed by name (contract in `particleman/scheduling/job_env.py`). `--dry-run` prints the command and env. `srun` is rejected under PBS and `mpiexec` under Slurm.
- HPO: `scripts/hpo.py` (Optuna, SQLite study). `scripts/submit_hpo.sh N` launches N PBS workers that share one study.
- `scripts/eval_predictions.py ++checkpoint=... ++out_file=x.npz ++eval_split=val` writes per-masked-particle predictions to an .npz file.
- Data tools: `explore_root_file.py` / `explore_h5_file.py` (find branch names and suggest config snippets), `convert_xaod_to_h5.py`, `demo_h5_format.py`. See `scripts/README.md`.

## Architecture

**Config layer.** `configs/config.yaml` composes the groups `data`, `model`, `training`, `output` and `distributed`. Each group file has a `# @package <group>` header. `src/particleman/config.py` holds matching dataclasses (`TrainConfig`). It is registered as `config_schema` but is *not* in the defaults list, so it doesn't validate anything. The YAML files are the real source of truth and the dataclass can fall behind them. When you add a config key, update the YAML defaults, the dataclass, and the place that reads it. `scripts/train.py` reads `cfg.data` with `.get()` and builds `ParticleConfig` by hand from `cfg.model`/`cfg.training`/`cfg.data.preprocessing`.

**Data (`src/particleman/data/`).** `BaseParticleLoader` (abstract) does the shared per-event pipeline: it merges all enabled `collections` into one particle list, maps raw IDs through `particle_id_map` (with `default_particle_id`, 15 = other/padding), applies the `preprocessing` cuts and `pt_scale`, optionally shuffles, then pads or truncates to `max_particles` and builds a bool `mask`. Event labels come from fnmatch-ing file basenames against `file_event_label_map`. Files with no match get -1 and are excluded from the classification loss. `ROOTParticleLoader` (uproot) and `HDF5ParticleLoader` implement file opening and column reads. `source.type` picks between them. `ParticleDataset` wraps a loader and returns tensors `pt, eta, phi, particle_id, mask, n_particles, event_label`. `create_dataloaders` / `create_datasets` do the train/val/test split from `data.split`.

**Model (`models/particle_transformer.py`).** `ParticleConfig` dataclass + enums `EmbeddingType` (JOINT/CONCAT) and `PhiEncoding` (RAW/SINCOS/NONE). `ParticleTransformer` normalizes features using fixed ranges, embeds them, runs a transformer encoder, and has per-feature regression and ID heads plus an optional event-label head (`n_event_types > 0`). `create_masks` and `compute_loss` live on the model. `AttentionBiasTransformer` (selected by `model.angular_attention_bias=true`) adds a learned pairwise Δη/Δφ attention bias (`AngularAttentionBias`) through custom encoder layers.

**Masking semantics (see MASKING.txt).** Masking works per particle. A masked particle gets Gaussian noise (std 0.1) on pt/eta/phi and its ID is replaced by a mask token (the last ID in the vocabulary). Padding is never masked. Loss counts only masked real particles, and the continuous targets are normalized. No mask indicator is fed to the embedding.

**Training (`training/`).** `ParticleTrainer` handles `mode='pretrain'` (masked prediction, retries mask creation if a batch has no masked particles) and `'classify'` (supervised event label; DDP uses `find_unused_parameters`). It runs AdamW, gradient accumulation, an optional torch.profiler, checkpointing and `pretrained_checkpoint` loading. `distributed.py` holds the DDP setup and rank helpers. `setup_distributed` gets rank info from `detect_rank_info`, which reads launcher env vars in order: torchrun, Slurm srun, MPICH/PALS mpiexec, OpenMPI. `MASTER_ADDR`/`MASTER_PORT` come from the launcher (`launch.sh` exports them). The device is `cuda:<local_rank % visible devices>`. Only rank 0 creates the `OutputManager`, the experiment logger and the logs. `OutputManager` writes to `outputs/<experiment_name>/<run_name>/{checkpoints,logs}`. `run_name` is the timestamp (from `output.timestamp`, set by submit.py) plus `_pbs<jobid>`/`_slurm<id>`. Hydra's own output dir and chdir are turned off.

**Job submission (`scheduling/`, `scripts/jobs/`).** `PBSScheduler`/`SlurmScheduler` build the submit command from the `cluster`/`resources`/`job` configs. `scripts/jobs/launch.sh` is the shared job script. It sources `scheduler_{pbs,slurm,none}.sh`, which define `sched_detect`, `sched_is_remote`, `sched_per_node_prefix` and `sched_per_gpu_prefix`. Multi-node torchrun is started once per node (mpiexec/srun) with c10d rendezvous. In batch mode the run dir `<output_dir>/<experiment>/<timestamp>_<scheduler><jobid>` gets `submit_command.sh`, `launch_command.sh` and `job_output.log`. HPO still uses its own `scripts/submit_hpo.sh` + `scripts/pbs/submit_hpo.pbs` (PBS only).

**Loggers (`loggers/`).** `create_logger(output.experiment_logger, ...)` returns an `MLFlowLogger`, `CometLogger` or `NoOpLogger`, all behind the `BaseLogger` interface (`log_params`, `log_metrics`, `close`).

## Conventions (from .cursorrules)

Type hints on all functions, Google/NumPy docstrings, dataclasses/enums for structured options, `logging` rather than print in library code, pathlib, and no hard-coded paths in `src/` (data paths belong in `configs/data/*.yaml`). ROADMAP.md lists planned work (per-feature masking with mask-indicator input, truth/reco flag, event-level features).
