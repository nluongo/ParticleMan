"""Tests for the trainer's throughput/timing record."""

import json
import time
from pathlib import Path

from particleman.loggers.base import NoOpLogger
from particleman.training.trainer import ParticleTrainer


def _bare_trainer(tmp_path: Path, world_size: int = 1) -> ParticleTrainer:
    """A ParticleTrainer with only the attributes the timing code uses."""
    trainer = ParticleTrainer.__new__(ParticleTrainer)
    trainer.is_main = True
    trainer.world_size = world_size
    trainer.step = 0
    trainer.experiment_logger = NoOpLogger()
    trainer.timing_path = tmp_path / "timing.json"
    trainer._train_start = time.time()
    trainer.timing = {
        "start_time": ParticleTrainer._now_iso(),
        "end_time": None,
        "elapsed_s": 0.0,
        "world_size": world_size,
        "initial_eval_s": None,
        "mean_it_per_sec": None,
        "epochs": [],
    }
    return trainer


def test_epoch_timing_written_to_file(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path, world_size=2)
    trainer._record_epoch_timing(0, "t0", n_batches=100, n_samples=3200, train_s=50.0, val_s=5.0)
    trainer._record_epoch_timing(1, "t1", n_batches=100, n_samples=3200, train_s=25.0, val_s=5.0)

    record = json.loads((tmp_path / "timing.json").read_text())
    assert [e["it_per_sec"] for e in record["epochs"]] == [2.0, 4.0]
    # samples/s is global: per-rank samples x world_size
    assert record["epochs"][0]["samples_per_sec"] == 128.0
    # Mean over the whole run is total batches / total train time, not the mean of rates
    assert record["mean_it_per_sec"] == round(200 / 75.0, 3)
    assert record["start_time"] is not None
    assert record["elapsed_s"] >= 0


def test_zero_train_time_does_not_divide_by_zero(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    trainer._record_epoch_timing(0, "t0", n_batches=0, n_samples=0, train_s=0.0, val_s=0.0)
    assert trainer.timing["epochs"][0]["it_per_sec"] == 0.0
    assert trainer.timing["mean_it_per_sec"] == 0.0


def test_non_main_process_writes_nothing(tmp_path: Path) -> None:
    trainer = _bare_trainer(tmp_path)
    trainer.is_main = False
    trainer._record_epoch_timing(0, "t0", n_batches=10, n_samples=10, train_s=1.0, val_s=0.0)
    assert not (tmp_path / "timing.json").exists()
