# This file contains the learning rate scheduler used in the training of the model.
from typing import List
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR,
    LRScheduler,
)
from experiment.parameters import ExperimentParams, LRSchedulerConfig


class Scheduler:
    def __init__(self, optimizer, exp_params: ExperimentParams, steps_per_epoch: int):
        self.optimizer = optimizer
        self.lrs_config: LRSchedulerConfig = exp_params.lrs_config
        self.scheduler: LRScheduler = None
        self.per_batch_steps: bool = False
        self.steps_per_epoch: int = steps_per_epoch
        self.warmup_steps: int = self.lrs_config.warmup_steps
        self.warmup: bool = self.warmup_steps is not None
        self.lr_history: List = []
        if exp_params.lrs_config is not None:
            if self.warmup:
                self.initialise_warmup_vars()
            self.create_scheduler()

    def initialise_warmup_vars(self):
        assert (
            self.lrs_config.scheduler_type != "ReduceLROnPlateau"
        ), "Warmup is not supported with ReduceLROnPlateau scheduler."
        self.per_batch_steps = True
        self.last_step = 0
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def create_scheduler(self):
        # if warmup, steps are counted in batches. Else in epochs.
        if self.lrs_config.scheduler_type == "StepLR":
            step_size = (
                self.lrs_config.step_size * self.steps_per_epoch
                if self.per_batch_steps
                else self.lrs_config.step_size
            )
            self.scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=self.lrs_config.gamma,
                verbose=True,
            )
        elif self.lrs_config.scheduler_type == "ExponentialLR":
            self.scheduler = ExponentialLR(
                self.optimizer, gamma=self.lrs_config.gamma, verbose=True
            )
        elif self.lrs_config.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.lrs_config.gamma,
                patience=self.lrs_config.patience,
                verbose=True,
            )
        elif self.lrs_config.scheduler_type == "CosineAnnealingLR":
            if self.lrs_config.T_max is None:
                self.per_batch_steps = True
                _Tmax = self.steps_per_epoch  # per batch annealing
            else:
                _Tmax = (
                    (self.lrs_config.T_max * self.steps_per_epoch - self.warmup_steps)
                    if self.per_batch_steps
                    else self.lrs_config.T_max
                )
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=_Tmax,
                eta_min=self.lrs_config.eta_min,
                last_epoch=self.lrs_config.last_epoch,
                verbose=False,
            )
        elif self.lrs_config.scheduler_type == "CosineAnnealingWarmRestarts":
            if self.lrs_config.T_0 is None:
                self.per_batch_steps = True  # per batch annealing
                _T0 = self.steps_per_epoch
            else:
                _T0 = (
                    (self.lrs_config.T_0 * self.steps_per_epoch - self.warmup_steps)
                    if self.per_batch_steps
                    else self.lrs_config.T_0
                )
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=_T0,
                T_mult=self.lrs_config.T_mult,
                eta_min=self.lrs_config.eta_min,
                last_epoch=self.lrs_config.last_epoch,
                verbose=False,
            )
        else:
            raise ValueError(
                f"Unsupported scheduler type: {self.lrs_config.scheduler_type}"
            )

    def reset_lr_history(self):
        self.lr_history = []

    def update_lr_history(self):
        self.lr_history.append(self.optimizer.param_groups[0]["lr"])

    def dampen(self):
        self.last_step += 1
        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = lr * (self.last_step / self.warmup_steps)

    def step(self, val_loss=None):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def per_batch_step(self):
        if self.per_batch_steps:
            if self.warmup:
                if self.last_step < self.warmup_steps:
                    self.dampen()
                else:
                    # Will only be called once to mark end of warmup period
                    self.warmup = False
                    self.step()
            else:
                self.step()
            # For debugging, can log lr each batch
            # self.update_lr_history()

    def per_epoch_step(self, val_loss):
        if not self.per_batch_steps:
            self.step(val_loss)
        self.update_lr_history
