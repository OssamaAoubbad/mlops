import datetime
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from madewithml import data, utils
from madewithml.config import logger, mlflow
from madewithml.models import FinetunedLLM

# Initialize Typer CLI app
app = typer.Typer()


def train_step(
    loader: DataLoader,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:  # pragma: no cover, tested via train workload
    """Train step.

    Args:
        ds (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        num_classes (int): number of classes.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.
        optimizer (torch.optimizer.Optimizer): optimizer to use for updating the model's weights.

    Returns:
        float: cumulative loss for the dataset.
    """
    model.train()
    loss = 0.0
    for i, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss


def eval_step(
    loader: DataLoader,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    device: torch.device,
) -> Tuple[float, np.array, np.array]:  # pragma: no cover, tested via train workload
    """Eval step.

    Args:
        ds (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        num_classes (int): number of classes.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.

    Returns:
        Tuple[float, np.array, np.array]: cumulative loss, ground truths and predictions.
    """
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict) -> None:  # pragma: no cover, tested via train workload
    """Compatibility wrapper for legacy entrypoints."""
    train_model(
        experiment_name=config.get("experiment_name", "mlops-project"),
        dataset_loc=config.get("dataset_loc", "datasets/dataset.csv"),
        train_loop_config=json.dumps(config.get("train_loop_config", {})),
        num_workers=config.get("num_workers", 1),
        cpu_per_worker=config.get("cpu_per_worker", 1),
        gpu_per_worker=config.get("gpu_per_worker", 0),
        num_samples=config.get("num_samples", 100),
        num_epochs=config.get("num_epochs", 10),
        batch_size=config.get("batch_size", 8),
        results_fp=config.get("results_fp", "results.json"),
    )


@app.command()
def train_model(
    experiment_name: str = "mlops-project",
    dataset_loc: str = "datasets/dataset.csv",
    train_loop_config: str = '{"dropout_p":0.3,"lr":1e-5,"lr_factor":0.8,"lr_patience":3}',
    num_workers: int = 1,
    cpu_per_worker: int = 1,
    gpu_per_worker: int = 0,
    num_samples: int = 100,
    num_epochs: int = 1,
    batch_size: int = 8,
    results_fp: str = "results.json",
) -> Dict:
    """Main train function to train our model.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        train_loop_config (str): arguments to use for training.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save results to. Defaults to None.

    Returns:
        Dict: training results.
    """
    # Set up
    train_loop_config = json.loads(train_loop_config)
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    # Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds["tag"].unique()
    train_loop_config["num_classes"] = len(tags)

    # Preprocess
    preprocessor = data.CustomPreprocessor()
    preprocessor = preprocessor.fit(train_ds)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)

    train_dataset = data.TextDataset(train_ds)
    val_dataset = data.TextDataset(val_ds)

    input_dim = len(preprocessor.vectorizer.get_feature_names_out())
    model = FinetunedLLM(
        input_dim=input_dim,
        num_classes=train_loop_config["num_classes"],
        dropout_p=train_loop_config["dropout_p"],
    )
    device = utils.get_device()
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_loop_config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_loop_config["lr_factor"],
        patience=train_loop_config["lr_patience"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=utils.collate_fn)

    history = []
    best_val_loss = float("inf")

    mlflow.set_experiment(experiment_name)
    run_name = f"{experiment_name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(train_loop_config)
        mlflow.log_param("dataset_loc", dataset_loc)

        for epoch in range(num_epochs):
            train_loss = train_step(train_loader, model, train_loop_config["num_classes"], loss_fn, optimizer, device)
            val_loss, _, _ = eval_step(val_loader, model, train_loop_config["num_classes"], loss_fn, device)
            scheduler.step(val_loss)

            metrics = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            history.append(metrics)
            mlflow.log_metrics(metrics, step=epoch)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss

        mlflow.pytorch.log_model(model, artifact_path="model")

        with tempfile.TemporaryDirectory() as tmp_dir:
            preprocessor_path = Path(tmp_dir, "preprocessor.pkl")
            preprocessor.save(str(preprocessor_path))
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        results = {
            "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
            "run_id": run.info.run_id,
            "params": train_loop_config,
            "best_val_loss": best_val_loss,
            "metrics": history,
        }

    logger.info(json.dumps(results, indent=2))
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(results, results_fp)
    return results


if __name__ == "__main__":  # pragma: no cover, application
    app()
