import datetime
import json
import itertools
from typing import Dict, Iterable

import typer
from typing_extensions import Annotated

from madewithml import train, utils
from madewithml.config import logger

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def tune_models(
    experiment_name: Annotated[str, typer.Option(help="name of the experiment for this training workload.")] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")] = None,
    initial_params: Annotated[str, typer.Option(help="initial config for the tuning workload.")] = None,
    num_runs: Annotated[int, typer.Option(help="number of runs in this tuning experiment.")] = 1,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    results_fp: Annotated[str, typer.Option(help="filepath to save results to.")] = None,
) -> Dict:
    """Hyperparameter tuning experiment.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        initial_params (str): initial config for the tuning workload.
        num_runs (int, optional): number of runs in this tuning experiment. Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save the tuning results. Defaults to None.

    Returns:
        Dict: results of the tuning experiment.
    """
    # Set up
    utils.set_seeds()
    base_config = json.loads(initial_params) if initial_params else {}
    base_config["num_samples"] = num_samples
    base_config["num_epochs"] = num_epochs
    base_config["batch_size"] = batch_size

    def _as_list(value, default: Iterable):
        if value is None:
            return list(default)
        if isinstance(value, list):
            return value
        return [value]

    param_grid = {
        "dropout_p": _as_list(base_config.get("dropout_p"), [0.3, 0.5, 0.7]),
        "lr": _as_list(base_config.get("lr"), [1e-5, 1e-4, 5e-4]),
        "lr_factor": _as_list(base_config.get("lr_factor"), [0.1, 0.5, 0.9]),
        "lr_patience": _as_list(base_config.get("lr_patience"), [1, 3, 5]),
    }

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    run_results = []
    best_result = None
    for i, values in enumerate(itertools.product(*param_values)):
        if num_runs and i >= num_runs:
            break
        config = dict(base_config)
        config.update(dict(zip(param_names, values)))
        result = train.train_model(
            experiment_name=experiment_name,
            dataset_loc=dataset_loc,
            train_loop_config=json.dumps(config),
            num_samples=config["num_samples"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            results_fp=None,
        )
        run_results.append(result)
        if best_result is None or result["best_val_loss"] < best_result["best_val_loss"]:
            best_result = result

    summary = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "best_run_id": best_result["run_id"] if best_result else None,
        "best_params": best_result["params"] if best_result else None,
        "best_val_loss": best_result["best_val_loss"] if best_result else None,
        "runs": run_results,
    }
    logger.info(json.dumps(summary, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(summary, results_fp)
    return summary


if __name__ == "__main__":  # pragma: no cover, application
    app()
