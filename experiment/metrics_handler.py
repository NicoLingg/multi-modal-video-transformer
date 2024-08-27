# This file contains the MetricsHandler class, which is used to save, load, and plot metrics for a model during training.
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from experiment.parameters import ExperimentParams
import names


class MetricsHandler:
    def __init__(
        self,
        checkpoint_dir,
        loss_weights,
        classification_names=None,
        regression_names=None,
    ):
        self.checkpoint_dir = checkpoint_dir
        metrics_list = ["lr"]
        for metric in "loss", "accuracy", "corrs", "components", "mae":
            metrics_list.extend([f"train_{metric}", f"val_{metric}"])

        self.metrics = {metric: {} for metric in metrics_list}
        self.classification_names = classification_names
        self.regression_names = regression_names
        self.component_names = classification_names + regression_names
        self.loss_weights = loss_weights

    def save(self):
        with open(os.path.join(self.checkpoint_dir, names.METRICS), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def load(self):
        with open(os.path.join(self.checkpoint_dir, names.METRICS), "r") as f:
            self.metrics = json.load(f)

    def print_epoch_metrics(self, epoch, num_epochs):
        epoch_str = str(epoch)
        if all(epoch_str in metric_dict for metric_dict in self.metrics.values()):
            verbose_metrics = ["train_loss", "val_loss"]
            metrics_output = ", ".join(
                f"{metric}: {values[epoch_str]:.4f}"
                for metric, values in self.metrics.items()
                if metric in verbose_metrics
            )

            print(f"Epoch [{epoch+1}/{num_epochs}], {metrics_output}")
        else:
            print(f"Metrics for epoch {epoch+1} have not been saved yet.")

    def add_epoch_metrics(
        self, epoch, epoch_metrics, num_epochs, print=True, save=True, plot_metrics=True
    ):
        for metric, values in epoch_metrics.items():
            if metric in self.metrics:
                self.metrics[metric][str(epoch)] = values
        if print:
            self.print_epoch_metrics(epoch, num_epochs)
        if save:
            self.save()
        if plot_metrics:
            self._plot_loss()
            self._plot_lr()
            self._plot_loss_components()
            self._plot_classification_accuracy()
            self._plot_regression_correlation()
            self._plot_regression_mae()

    def _plot_lr(self):
        fig, ax = plt.subplots()
        lr_values = list(self.metrics["lr"].values())
        if all(isinstance(i, list) for i in lr_values):
            lr_values = [item for sublist in lr_values for item in sublist]
            epochs = [
                epoch + i / len(sublist)
                for epoch, sublist in enumerate(self.metrics["lr"].values())
                for i in range(len(sublist))
            ]
        else:
            epochs = list(range(len(lr_values)))
        ax.plot(epochs, lr_values, label="learning rate", linewidth=3)
        ax.scatter(epochs, lr_values)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Learning Rate", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, "lr.png"))
        plt.close()

    def _plot_loss(self):
        fig, ax1 = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("white")

        for metric in ["train_loss", "val_loss"]:
            values = self.metrics[metric]
            epochs = [int(epoch) for epoch in values.keys()]
            metric_values = list(values.values())
            ax1.plot(epochs, metric_values, label=metric, linewidth=3)
            ax1.scatter(epochs, metric_values)
        ax1.set_xlabel("Epoch", fontsize=14)
        ax1.set_ylabel("Loss", fontsize=14)
        ax1.set_title("Training and Validation Loss Over Epochs", fontsize=16)
        ax1.legend(title="Metrics", loc="upper right", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, "loss.png"))
        plt.close()

    def _plot_classification_accuracy(self):
        epochs = [int(epoch) for epoch in self.metrics["train_accuracy"].keys()]
        accuracy_dict = {
            "train": list(self.metrics["train_accuracy"].values()),
            "val": list(self.metrics["val_accuracy"].values()),
        }

        for i, name in enumerate(self.classification_names):
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("white")

            for train_or_val, all_epoch_accuracy in accuracy_dict.items():
                values = [epoch_accuracy[i] for epoch_accuracy in all_epoch_accuracy]
                ax.plot(epochs, values, label=train_or_val, linewidth=3)
                ax.scatter(epochs, values)
            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("Accuracy", fontsize=14)
            ax.set_title(f"Training and Validation Accuracy for {name}", fontsize=16)
            ax.legend(loc="upper right", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.checkpoint_dir, f"accuracy_{name}.png"))
            plt.close()

    def _plot_regression_correlation(self):
        epochs = [int(epoch) for epoch in self.metrics["train_corrs"].keys()]
        corrs_dict = {
            "train": list(self.metrics["train_corrs"].values()),
            "val": list(self.metrics["val_corrs"].values()),
        }

        for i, name in enumerate(self.regression_names):
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("white")

            for train_or_val, all_epoch_corrs in corrs_dict.items():
                values = [epoch_corr[i] for epoch_corr in all_epoch_corrs]
                ax.plot(epochs, values, label=train_or_val, linewidth=3)
                ax.scatter(epochs, values)
            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("Correlation", fontsize=14)
            ax.set_title(f"Training and Validation Correlation for {name}", fontsize=16)
            ax.legend(loc="upper right", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.checkpoint_dir, f"correlation_{name}.png"))
            plt.close()

    def _plot_regression_mae(self):
        epochs = [int(epoch) for epoch in self.metrics["train_mae"].keys()]
        mae_dict = {
            "train": list(self.metrics["train_mae"].values()),
            "val": list(self.metrics["val_mae"].values()),
        }

        for i, name in enumerate(self.regression_names):
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor("white")

            for train_or_val, all_epoch_mae in mae_dict.items():
                values = [epoch_mae[i] for epoch_mae in all_epoch_mae]
                ax.plot(epochs, values, label=train_or_val, linewidth=3)
                ax.scatter(epochs, values)
            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("MAE", fontsize=14)
            ax.set_title(f"Training and Validation MAE for {name}", fontsize=16)
            ax.legend(loc="upper right", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.checkpoint_dir, f"mae_{name}.png"))
            plt.close()

    def _plot_loss_components(self):
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        fig.patch.set_facecolor("white")

        for i, metric_name in enumerate(["train_components", "val_components"]):
            epochs = [int(epoch) for epoch in self.metrics[metric_name].keys()]
            all_epoch_metrics = list(self.metrics[metric_name].values())
            for j, name in enumerate(self.component_names):
                component = [epoch_metrics[j] for epoch_metrics in all_epoch_metrics]
                axs[i].plot(epochs, component, label=name, linewidth=3)
                axs[i].scatter(epochs, component)
            axs[i].set_xlabel("Epoch", fontsize=14)
            axs[i].set_ylabel("Loss", fontsize=14)
            train_or_val = metric_name.split("_")[0].capitalize()
            axs[i].set_title(f"{train_or_val} Loss Components Over Epochs", fontsize=16)
            axs[i].legend(title="Metrics", loc="upper right", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, "loss_components.png"))
        plt.close()

    def generate_report(self, sigma=1):
        report = {}
        for i, name in enumerate(self.classification_names):
            stats = {}
            for train_or_val in "val", "train":
                values = [
                    value[i]
                    for value in self.metrics[f"{train_or_val}_accuracy"].values()
                ]
                values_smooth = gaussian_filter1d(values, sigma=sigma, mode="nearest")
                stats[f"{train_or_val}_best"] = max(values_smooth)
                stats[f"{train_or_val}_epoch"] = np.argmax(values_smooth)
                stats[f"{train_or_val}_last"] = values_smooth[-1]
            report[name] = stats

        regression_start = len(self.classification_names)
        for i in range(regression_start, len(self.component_names), 1):
            name = self.component_names[i]
            loss_weight = self.loss_weights[i]
            stats = {}
            for metric in "mae", "components":
                for train_or_val in "val", "train":
                    label = ""
                    if metric == "components":
                        values = [
                            value[i]
                            for value in self.metrics[
                                f"{train_or_val}_{metric}"
                            ].values()
                        ]
                        values = [np.sqrt(value / loss_weight) for value in values]
                        label = "rmse_"
                    else:
                        values = [
                            value[i - regression_start]
                            for value in self.metrics[
                                f"{train_or_val}_{metric}"
                            ].values()
                        ]
                    values_smooth = gaussian_filter1d(
                        values, sigma=sigma, mode="nearest"
                    )
                    stats[f"{train_or_val}_{label}best"] = min(values_smooth)
                    stats[f"{train_or_val}_{label}epoch"] = np.argmin(values_smooth)
                    stats[f"{train_or_val}_{label}last"] = values_smooth[-1]
            report[name] = stats

        report_df = pd.DataFrame(report).T
        report_df.to_csv(os.path.join(self.checkpoint_dir, names.REPORT))

    @classmethod
    def load_from_subfolder(cls, sub_folder):
        exp = ExperimentParams.load_from_subfolder(sub_folder)
        return cls.load_from_exp_params(exp)

    @classmethod
    def load_from_exp_params(cls, params: ExperimentParams):
        unique_model_name = f"{params.model_name}_{params.run_id}"
        checkpoint_dir = os.path.join(names.MODELS_DIR, unique_model_name)
        if not os.path.exists(checkpoint_dir):
            raise Exception(f"Directory does not exist: {checkpoint_dir}")
        instance = cls(
            checkpoint_dir,
            classification_names=list(params.classification_tasks.keys()),
            regression_names=params.regression_tasks,
            loss_weights=params.loss_weights,
        )
        instance.load()
        return instance
