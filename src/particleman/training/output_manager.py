"""
Output manager for organizing training outputs.

Creates a unified directory structure:
    <base_dir>/<experiment_name>/<run_name>/
    ├── checkpoints/       # Model checkpoints
    ├── logs/              # Additional log files
    ├── config.json        # Training configuration
    ├── job_info.json      # PBS/SLURM job info (if applicable)
    ├── output.log         # Training log
    └── summary.json       # Final metrics
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Manages training output directories and files.
    
    Uses experiment_name/run_name structure consistent with MLflow.
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        base_dir: str = "outputs",
    ) -> None:
        """
        Initialize the output manager.
        
        Args:
            experiment_name: Name of the experiment (groups related runs).
            run_name: Name for this run. If None, auto-generated from timestamp.
            base_dir: Base directory for all outputs.
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        
        # Generate run name if not provided (timestamp, same as used for MLflow)
        if run_name is None:
            run_name = self._generate_run_name()
        self.run_name = run_name
        
        # Set up directory structure
        self.experiment_dir = self.base_dir / experiment_name
        self.run_dir = self.experiment_dir / run_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.log_dir = self.run_dir / "logs"
        
        self._create_directories()
    
    def _generate_run_name(self) -> str:
        """
        Generate a unique run name.
        
        Uses Unix timestamp for consistency with MLflow run_name.
        Appends PBS/SLURM job ID if available.
        """
        timestamp = str(round(time.time()))
        
        # Append job ID if in batch environment
        pbs_jobid = os.environ.get("PBS_JOBID")
        if pbs_jobid:
            job_num = pbs_jobid.split(".")[0]
            return f"{timestamp}_pbs{job_num}"
        
        slurm_jobid = os.environ.get("SLURM_JOB_ID")
        if slurm_jobid:
            return f"{timestamp}_slurm{slurm_jobid}"
        
        return timestamp
    
    def _create_directories(self) -> None:
        """Create all output directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.run_dir}")
    
    def setup_logging(
        self,
        level: int = logging.INFO,
        log_filename: str = "output.log",
    ) -> Path:
        """
        Set up logging to file and console.
        
        Args:
            level: Logging level.
            log_filename: Name of the log file.
        
        Returns:
            Path to the log file.
        """
        log_path = self.run_dir / log_filename
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_path}")
        return log_path
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.json") -> Path:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary to save.
            filename: Name of the config file.
        
        Returns:
            Path to the saved config file.
        """
        filepath = self.run_dir / filename
        
        # Convert non-serializable types
        def convert(obj):
            if hasattr(obj, "value"):  # Enum
                return obj.value
            if hasattr(obj, "__dict__"):  # Dataclass or object
                return {k: convert(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        serializable_config = convert(config)
        
        with open(filepath, "w") as f:
            json.dump(serializable_config, f, indent=2, default=str)
        
        logger.info(f"Saved config to {filepath}")
        return filepath
    
    def save_job_info(self) -> Optional[Path]:
        """
        Save job scheduler information (PBS/SLURM) if available.
        
        Returns:
            Path to the saved job info file, or None if not in a job.
        """
        job_info = {}
        
        # PBS environment variables
        pbs_vars = [
            "PBS_JOBID", "PBS_JOBNAME", "PBS_QUEUE", "PBS_O_WORKDIR",
            "PBS_NODEFILE", "PBS_O_HOST",
        ]
        for var in pbs_vars:
            if var in os.environ:
                job_info[var] = os.environ[var]
        
        # SLURM environment variables
        slurm_vars = [
            "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_PARTITION",
            "SLURM_NNODES", "SLURM_NTASKS", "SLURM_NODELIST",
        ]
        for var in slurm_vars:
            if var in os.environ:
                job_info[var] = os.environ[var]
        
        if not job_info:
            return None
        
        # Add node list from PBS nodefile if available
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and Path(nodefile).exists():
            with open(nodefile) as f:
                job_info["nodes"] = list(set(line.strip() for line in f.readlines()))
        
        filepath = self.run_dir / "job_info.json"
        with open(filepath, "w") as f:
            json.dump(job_info, f, indent=2)
        
        logger.info(f"Saved job info to {filepath}")
        return filepath
    
    def save_summary(self, metrics: Dict[str, Any]) -> Path:
        """
        Save training summary/results.
        
        Args:
            metrics: Final training metrics to save.
        
        Returns:
            Path to the saved summary file.
        """
        summary = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "completed_at": datetime.now().isoformat(),
            "metrics": metrics,
        }
        
        filepath = self.run_dir / "summary.json"
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {filepath}")
        return filepath
    
    def __str__(self) -> str:
        return f"OutputManager({self.experiment_name}/{self.run_name})"
    
    def __repr__(self) -> str:
        return self.__str__()
