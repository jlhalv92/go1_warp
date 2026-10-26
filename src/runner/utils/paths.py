"""Keep outputs rooted at the repository, independently of the caller's cwd."""
from pathlib import Path
from src import ROOT


def rooted(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def output_paths(config):
    suffix = Path(config.experiment_name) / f'run_{config.run_id}'
    return rooted(config.output_dir) / suffix, rooted(config.log_dir) / suffix


def run_paths(config):
    run_dir, log_dir = output_paths(config)
    if not config.resume and (run_dir.exists() or log_dir.exists()):
        raise FileExistsError(f'Run already exists: {run_dir}. Choose another --run-id or use --resume.')
    run_dir.mkdir(parents=True, exist_ok=bool(config.resume))
    log_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, log_dir
