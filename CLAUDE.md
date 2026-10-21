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

- `python scripts/run.py [overrides]` is the main entry point. The default root config is `configs/config.yaml`. Pick a different root with `--config-name bbllv08` (or `mc20_ttbar`). Override with `group=name` or `key.path=value`, e.g. `data=bbllv08_classify training=quick_test training.mode=classify model.d_model=256`. `--cfg job` prints the resolved config. `job.mode` picks what it does (`particleman/scheduling/run_mode.py:resolve_mode`). `train` runs `particleman.training.run_training` in-process. `launch` runs `scripts/jobs/launch.sh scripts/train.py <overrides>`. `submit` runs `qsub`/`sbatch` on `launch.sh` (`particleman/scheduling/submit.py:submit_job`). `auto` (the default) picks as follows:
  - train under torchrun;
  - submit when `cluster.scheduler` isn't `none` and there's no `PBS_JOBID`/`SLURM_JOB_ID`;
  - launch inside a job or when `resources.nnodes`/`gpus_per_node` (null means count local GPUs) asks for more than one process;
  - train otherwise.

  Workers always run `scripts/train.py`, never `run.py`. They get the same overrides plus `output.timestamp`.
- Batch: `python scripts/run.py cluster=<site> resources=<size> [cluster.*=/resources.*=/job.*=...] <training overrides>`. Site settings are in `configs/cluster/` (the default `local` has no scheduler) and job sizes in `configs/resources/`. Submitting needs `resources.gpus_per_node` set. Settings reach the job as env vars, passed by name (contract in `particleman/scheduling/job_env.py`). `job.dry_run=true` prints the command and env. `srun` is rejected under PBS and `mpiexec` under Slurm.
- `python scripts/train.py [overrides]` is the barebones in-process trainer (what workers run). `bash scripts/jobs/launch.sh scripts/train.py <overrides>` and plain `torchrun --standalone --nproc_per_node=N scripts/train.py ...` still work. For launch.sh, `LAUNCHER=torchrun` (the default), `mpiexec` (PBS) or `srun` (Slurm) picks the launch mode.
- HPO: `scripts/hpo.py` (Optuna, SQLite study). `scripts/submit_hpo.sh N` launches N PBS workers that share one study.
- `scripts/eval_predictions.py ++checkpoint=... ++out_file=x.npz ++eval_split=val` writes per-masked-particle predictions to an .npz file.
- Data tools: `explore_root_file.py` / `explore_h5_file.py` (find branch names and suggest config snippets), `convert_xaod_to_h5.py`, `demo_h5_format.py`. See `scripts/README.md`.

## Architecture

**Config layer.** `configs/config.yaml` composes the groups `data`, `model`, `training`, `output`, `distributed`, `cluster`, `resources` and `job`. Each group file has a `# @package <group>` header. The YAML files hold the values. The dataclasses in `src/particleman/config.py` are strict schemas. `register_configs()` stores them in the ConfigStore as `base_<group>` and the root as `base_config`. Every group YAML starts with `defaults: [base_<group>, _self_]`, and every root YAML lists `base_config` first and `_self_` last. The composition order is: schema, group YAML, root YAML, CLI. An unknown key or a wrong type is an error, so a new key needs a dataclass field as well as the YAML default. New root-only keys (e.g. `++checkpoint` in `eval_predictions.py`, `+trials` in `hpo.py`) still need `+`/`++`. Each entry point calls `register_configs()` before composing. `particleman/training/pipeline.py` (`run_training`) reads `cfg.data` with `.get()` and builds `ParticleConfig` by hand (`build_model_config`) from `cfg.model`/`cfg.training`/`cfg.data.preprocessing`.

**Data (`src/particleman/data/`).** `BaseParticleLoader` (abstract) does the shared per-event pipeline: it merges all enabled `collections` into one particle list, maps raw IDs through `particle_id_map` (with `default_particle_id`, 15 = other/padding), applies the `preprocessing` cuts and `pt_scale`, optionally shuffles, then pads or truncates to `max_particles` and builds a bool `mask`. Event labels come from fnmatch-ing file basenames against `file_event_label_map`. Files with no match get -1 and are excluded from the classification loss. `ROOTParticleLoader` (uproot) and `HDF5ParticleLoader` implement file opening and column reads. `source.type` picks between them. `ParticleDataset` wraps a loader and returns tensors `pt, eta, phi, particle_id, mask, n_particles, event_label`. `create_dataloaders` / `create_datasets` do the train/val/test split from `data.split`.

**Model (`models/particle_transformer.py`).** `ParticleConfig` dataclass + enums `EmbeddingType` (JOINT/CONCAT) and `PhiEncoding` (RAW/SINCOS/NONE). `ParticleTransformer` normalizes features using fixed ranges, embeds them, runs a transformer encoder, and has per-feature regression and ID heads plus an optional event-label head (`n_event_types > 0`). `create_masks` and `compute_loss` live on the model. `AttentionBiasTransformer` (selected by `model.angular_attention_bias=true`) adds a learned pairwise Δη/Δφ attention bias (`AngularAttentionBias`) through custom encoder layers.

**Masking semantics (see MASKING.txt).** Masking works per particle. A masked particle gets Gaussian noise (std 0.1) on pt/eta/phi and its ID is replaced by a mask token (the last ID in the vocabulary). Padding is never masked. Loss counts only masked real particles, and the continuous targets are normalized. No mask indicator is fed to the embedding.

**Training (`training/`).** `ParticleTrainer` handles `mode='pretrain'` (masked prediction, retries mask creation if a batch has no masked particles) and `'classify'` (supervised event label; DDP uses `find_unused_parameters`). It runs AdamW, gradient accumulation, an optional torch.profiler, checkpointing and `pretrained_checkpoint` loading. `distributed.py` holds the DDP setup and rank helpers. `setup_distributed` gets rank info from `detect_rank_info`, which reads launcher env vars in order: torchrun, Slurm srun, MPICH/PALS mpiexec, OpenMPI. `MASTER_ADDR`/`MASTER_PORT` come from the launcher (`launch.sh` exports them). The device is `cuda:<local_rank % visible devices>`. Only rank 0 creates the `OutputManager`, the experiment logger and the logs. `OutputManager` writes to `outputs/<experiment_name>/<run_name>/{checkpoints,logs}`. The trainer rewrites `timing.json` there after every epoch: start time, elapsed time, and per-epoch it/s (same as the tqdm bar) and samples/s. `summary.json` repeats the start time, elapsed time and mean it/s. `run_name` is the timestamp (from `output.timestamp`, set by run.py) plus `_pbs<jobid>`/`_slurm<id>`. Hydra's own output dir and chdir are turned off.

**Job submission (`scheduling/`, `scripts/jobs/`).** `PBSScheduler`/`SlurmScheduler` build the submit command from the `cluster`/`resources`/`job` configs; `submit_job` drives them. `scripts/jobs/launch.sh` is the shared job script. It sources `scheduler_{pbs,slurm,none}.sh`, which define `sched_detect`, `sched_is_remote`, `sched_per_node_prefix` and `sched_per_gpu_prefix`. Multi-node torchrun is started once per node (mpiexec/srun) with c10d rendezvous. In batch mode the run dir `<output_dir>/<experiment>/<timestamp>_<scheduler><jobid>` gets `submit_command.sh`, `launch_command.sh` and `job_output.log`. HPO still uses its own `scripts/submit_hpo.sh` + `scripts/pbs/submit_hpo.pbs` (PBS only).

**Loggers (`loggers/`).** `create_logger(output.experiment_logger, ...)` returns an `MLFlowLogger`, `CometLogger` or `NoOpLogger`, all behind the `BaseLogger` interface (`log_params`, `log_metrics`, `close`).

## Conventions (from .cursorrules)

Type hints on all functions, Google/NumPy docstrings, dataclasses/enums for structured options, `logging` rather than print in library code, pathlib, and no hard-coded paths in `src/` (data paths belong in `configs/data/*.yaml`). ROADMAP.md lists planned work (per-feature masking with mask-indicator input, truth/reco flag, event-level features).
