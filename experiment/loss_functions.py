# This file contains the loss functions used in the training of the model.
import torch
from torch import nn
from experiment.parameters import ExperimentParams


def supervised_loss(
    outputs,
    labels,
    loss_weights,
    params: ExperimentParams,
    return_components=False,
    class_weight=None,
):
    # outputs.shape: (batch, target_length, predictions)
    outputs = outputs.flatten(start_dim=0, end_dim=1)
    labels = labels.flatten(start_dim=0, end_dim=1)

    weighted_loss_components = []
    n = 0  # index for the outputs tensor

    # Classification tasks
    for i, num_classes in enumerate(params.classification_tasks.values()):
        # Create a mask for labels >= 0. Labels < 0 are exluded from the loss calculation (e.g. no trust data)
        mask = labels[:, i].ge(0)
        if sum(mask) == 0:
            weighted_loss_components.append(
                torch.tensor(0.0, requires_grad=False, device=outputs.device)
            )
        else:
            if num_classes == 2:  # binary classification
                weight = class_weight[n] if class_weight is not None else None
                loss_class = nn.BCEWithLogitsLoss(
                    reduction="mean", pos_weight=weight
                )  # can add for class balancing
                weighted_loss_components.append(
                    loss_weights[i]
                    * loss_class(
                        input=outputs[mask, n], target=labels[mask, i].to(torch.float32)
                    )
                )
            else:  # multi-class classification
                loss_class = nn.CrossEntropyLoss(
                    reduction="mean"
                )  # no class balancing for now
                weighted_loss_components.append(
                    loss_weights[i]
                    * loss_class(
                        input=outputs[mask, n : n + num_classes],
                        target=labels[mask, i].long(),
                    )
                )
        n = (n + num_classes) if num_classes > 2 else (n + 1)

    # Regression tasks
    n_class_tasks = len(params.classification_tasks)
    regression_labels = labels[:, n_class_tasks:]
    regression_outputs = outputs[:, n:]
    for i in range(len(params.regression_tasks)):
        loss_mse = nn.MSELoss(reduction="mean")
        weighted_loss_components.append(
            loss_weights[n_class_tasks + i]
            * loss_mse(input=regression_outputs[:, i], target=regression_labels[:, i])
        )

    # Combine losses
    total_loss = sum(weighted_loss_components)

    # Return loss components if specified (for loss visualization and debugging purposes)
    if return_components:
        return total_loss, weighted_loss_components
    else:
        return total_loss
