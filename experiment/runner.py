# This file contains core pipeline for running experiments and training models.
from typing import List
import torch
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
from tqdm import tqdm
import gc
import random
from experiment.lr_scheduler import Scheduler
from experiment.parameters import ExperimentParams
from experiment.metrics_handler import MetricsHandler
from experiment.loss_functions import supervised_loss
from data.sampler import VideoSampler
from models.video_model import VideoModel
import names


class Experiment:
    def __init__(self, exp_params: ExperimentParams):
        # Set seed
        random.seed(exp_params.seed)
        np.random.seed(exp_params.seed)
        torch.manual_seed(exp_params.seed)

        # Set-up experiment
        self.exp_params = exp_params
        self.num_epochs = exp_params.num_epochs
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using { self.device} device")
        if exp_params.enable_amp and self.device.type != "cuda":
            print(
                "Warning: AMP is enabled but CUDA is not available. AMP will be disabled."
            )
        self.enable_amp = exp_params.enable_amp and self.device.type == "cuda"

        model = VideoModel(exp_params).to(self.device)
        if self.device.type == "cuda":
            print("Compiling model...")
            self.model = torch.compile(model)
        else:
            self.model = model

        # Sampling & data
        sampler = VideoSampler(exp_params)
        self.train_loader, self.validation_loader, class_weight = (
            sampler.create_dataloaders(return_class_weight=True)
        )
        self.class_weight = (
            torch.tensor(class_weight, dtype=torch.float32, device=self.device)
            if exp_params.enable_class_balancing
            else None
        )

        # Optimizer, scheduler & loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=exp_params.lr,
            weight_decay=exp_params.weight_decay,
        )
        self.scheduler = Scheduler(
            self.optimizer, exp_params, steps_per_epoch=len(self.train_loader)
        )
        self.loss_weights = torch.tensor(exp_params.loss_weights).to(self.device)

        # Tracking & metrics
        self.metrics_handler = MetricsHandler(
            self.experiment_dir,
            classification_names=list(self.exp_params.classification_tasks.keys()),
            regression_names=self.exp_params.regression_tasks,
            loss_weights=self.exp_params.loss_weights,
        )
        self.n_classification_tasks = len(exp_params.classification_tasks)
        self.n_regression_tasks = len(exp_params.regression_tasks)
        self.n_tasks = self.n_classification_tasks + self.n_regression_tasks

        # Save experiment parameters
        exp_params.save()

    @property
    def experiment_dir(self):
        unique_model_name = f"{self.exp_params.model_name}_{self.exp_params.run_id}"
        return os.path.join(names.MODELS_DIR, unique_model_name)

    def delete(self):
        del self.model
        del self.optimizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def run(self):
        raise NotImplementedError

    def save_model(self, name=names.BEST_MODEL):
        self.model.save(os.path.join(self.checkpoint_dir, name))

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError


class SupervisedExperiment(Experiment):
    def __init__(self, params: ExperimentParams):
        super().__init__(params)
        self.scaler = GradScaler() if self.enable_amp else None

    def run(self):
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            (
                train_loss,
                train_loss_components,
                train_accuracy,
                train_corrs,
                train_mae,
            ) = self.loop(train=True)
            val_loss, val_loss_components, val_accuracy, val_corrs, val_mae = self.loop(
                train=False
            )
            self.scheduler.per_epoch_step(val_loss)

            self.metrics_handler.add_epoch_metrics(
                epoch,
                {
                    "lr": (
                        self.scheduler.lr_history
                        if len(self.scheduler.lr_history) > 0
                        else self.optimizer.param_groups[0]["lr"]
                    ),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "train_corrs": train_corrs,
                    "val_corrs": val_corrs,
                    "train_components": train_loss_components,
                    "val_components": val_loss_components,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                },
                num_epochs=self.num_epochs,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save(names.BEST_MODEL)

            self.model.save(names.LAST_MODEL)

    def loop(self, train: bool):
        mode = "Training" if train else "Validating"
        loader = self.train_loader if train else self.validation_loader
        self.model.train() if train else self.model.eval()
        n_batches = len(loader)

        # Initialize accumulators for metrics
        if train:
            self.scheduler.reset_lr_history()
        total_loss = 0
        total_accuracy, accuracy_counts = [0] * self.n_classification_tasks, [
            0
        ] * self.n_classification_tasks
        total_corrs = [0] * self.n_regression_tasks
        total_mae = [0] * self.n_regression_tasks
        total_loss_components, loss_components_counts = [0] * self.n_tasks, [
            0
        ] * self.n_tasks

        for i, (inputs, labels) in enumerate(tqdm(loader, desc=mode)):
            images, labels = inputs["images"].to(self.device), labels.to(self.device)
            fusion_features = (
                inputs["fusion_features"].to(self.device)
                if "fusion_features" in inputs
                else None
            )

            if train:
                self.optimizer.zero_grad()
                # Forward pass with autocast if using AMP
                with autocast(enabled=self.enable_amp):
                    outputs = self.model(images=images, fusion_features=fusion_features)
                    loss, loss_components = supervised_loss(
                        outputs,
                        labels,
                        self.loss_weights,
                        self.exp_params,
                        return_components=True,
                        class_weight=self.class_weight,
                    )

                if loss.requires_grad:
                    # Backward and optimize with scaled loss if using AMP
                    if self.enable_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                self.scheduler.per_batch_step()
            else:
                with torch.no_grad():
                    # Forward pass with autocast if using AMP
                    with autocast(enabled=self.enable_amp):
                        outputs = self.model(images, fusion_features=fusion_features)
                        loss, loss_components = supervised_loss(
                            outputs,
                            labels,
                            self.loss_weights,
                            self.exp_params,
                            return_components=True,
                            class_weight=None,
                        )

            # Accumulate metrics
            total_loss += loss.item()
            for i, loss_component in enumerate(loss_components):
                if loss_component > 0:
                    total_loss_components[i] += loss_component.item()
                    loss_components_counts[i] += 1

            n = 0
            for i, num_classes in enumerate(
                self.exp_params.classification_tasks.values()
            ):
                mask = labels[:, :, i].ge(0)
                if sum(mask) > 0:
                    if num_classes == 2:
                        predictions = (
                            outputs[:, :, n] > 0
                        ).long()  # Predict class 1 if the output is greater than 0
                        accuracy = torch.mean(
                            (predictions == labels[:, :, i]).float()[mask]
                        )
                    else:
                        predictions = torch.argmax(
                            outputs[:, :, n : n + num_classes], dim=2
                        )
                        accuracy = torch.mean(
                            (predictions == labels[:, :, i]).float()[mask]
                        )
                    total_accuracy[i] += accuracy.item()
                    accuracy_counts[i] += 1
                n = (n + num_classes) if num_classes > 2 else (n + 1)

            for i in range(self.n_regression_tasks):
                predicted = outputs[:, :, n + i].flatten()
                actual = labels[:, :, self.n_classification_tasks + i].flatten()
                corr = torch.corrcoef(torch.stack([predicted, actual]))[0, 1]
                total_corrs[i] += corr.item()
                total_mae[i] += torch.mean(torch.abs(predicted - actual)).item()

                # For debugging
                # if torch.isnan(corr):
                #     print("The tensor 'corr' contains NaN values.")
                #     if torch.isnan(predicted).any():
                #         print("The tensor 'predicted' contains NaN values.")
                #     if torch.isnan(actual).any():
                #         print("The tensor 'actual' contains NaN values.")

        # Calculate average loss components
        avg_loss_components = [
            total / count if count > 0 else 0
            for total, count in zip(total_loss_components, loss_components_counts)
        ]
        avg_accuracy = [
            total / count if count > 0 else 0
            for total, count in zip(total_accuracy, accuracy_counts)
        ]
        avg_corrs = [total / n_batches for total in total_corrs]
        avg_mae = [total / n_batches for total in total_mae]

        return (
            total_loss / n_batches,
            avg_loss_components,
            avg_accuracy,
            avg_corrs,
            avg_mae,
        )


def run_experiment(exp_params: ExperimentParams, seed=0):
    experiment = SupervisedExperiment(exp_params)
    experiment.run()
    experiment.metrics_handler.generate_report()
    experiment.delete()
