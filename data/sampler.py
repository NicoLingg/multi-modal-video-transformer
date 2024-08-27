# This file contains the VideoSampler class, which is used to sample sequences from the dataset for training and testing.

from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Any, Dict
import dataclasses
import multiprocessing
import os
from torch.utils.data import DataLoader

from experiment.parameters import ExperimentParams
from data.dataset import CustomImageDataset
import names


class VideoSampler:
    def __init__(self, params: ExperimentParams):
        self.params = params
        self.file_path = os.path.join(names.DATASET_DIR, names.DATA)
        assert os.path.exists(
            self.file_path
        ), f"File {self.file_path} not found. Please download a dataset, or generate an illustrative one by running the generate_example_dataset.py script."

    def _load_data(self):
        """
        Function to load data from file.
        """
        df = pd.read_csv(self.file_path)
        df[names.IMAGE_PATH] = df.apply(
            lambda row: os.path.join(
                names.DATASET_DIR,
                row[names.UNIQUE_ID] + "_frames",
                row[names.IMAGE_PATH],
            ),
            axis=1,
        )
        return df

    def _generate_sequences(self, data, overrides: Dict[str, Any] = {}):
        if not overrides:
            params = self.params
        else:
            params = dataclasses.replace(self.params, **overrides)

        input_sequences = defaultdict(list)
        target_sequences = []
        extended_sequence_length = params.sequence_length * params.downsample_factor
        classification_targets = list(params.classification_tasks.keys())

        for start_idx in range(0, len(data) - extended_sequence_length, params.stride):
            inputs = {}
            end_idx = start_idx + extended_sequence_length
            downsampled_sequence = data.iloc[start_idx:end_idx][
                :: params.downsample_factor
            ]
            target_sequence = downsampled_sequence[-params.target_length :][
                classification_targets + params.regression_tasks
            ].values
            inputs[names.IMAGES] = downsampled_sequence[names.IMAGE_PATH].values
            if params.fusion_config:
                for feature in params.fusion_config.fusion_features:
                    inputs[feature] = downsampled_sequence[feature].values

            target_sequences.append(target_sequence)
            for name, sequence in inputs.items():
                input_sequences[name].append(sequence)

        return input_sequences, target_sequences

    def _process_data(
        self,
        data,
        unique_id,
        train_sequences,
        train_targets,
        test_sequences,
        test_targets,
        overrides,
    ):
        """
        Process data for a single unique ID.
        """
        sequences, target_sequences = self._generate_sequences(
            data, overrides=overrides
        )

        if unique_id in self.params.test_ids:
            test_targets.extend(target_sequences)
            for name, sequence in sequences.items():
                test_sequences[name].extend(sequence)
        else:
            train_targets.extend(target_sequences)
            for name, sequence in sequences.items():
                train_sequences[name].extend(sequence)

    def sample_train_test_sequences(self, overrides: Dict[str, Any] = {}):
        """
        Function to create training and test sequences with downsampling.
        """
        df = self._load_data()
        train_targets, test_targets = [], []
        train_sequences, test_sequences = defaultdict(list), defaultdict(list)

        for unique_id in df[names.UNIQUE_ID].unique():
            unique_id_data = df[df[names.UNIQUE_ID] == unique_id]
            self._process_data(
                unique_id_data,
                unique_id,
                train_sequences,
                train_targets,
                test_sequences,
                test_targets,
                overrides,
            )

        train_sequences = {
            name: np.array(sequence) for name, sequence in train_sequences.items()
        }
        test_sequences = {
            name: np.array(sequence) for name, sequence in test_sequences.items()
        }
        return (
            train_sequences,
            np.array(train_targets, dtype=np.float32),
            test_sequences,
            np.array(test_targets, dtype=np.float32),
        )

    def sample_inference_sequences(
        self, overrides: Dict[str, Any] = {}, unique_id_list=None
    ):
        # Override stride and target length parameter
        if "stride" not in overrides:
            overrides["stride"] = 1

        if "target_length" not in overrides:
            overrides["target_length"] = 1

        # Generate sequences and labels for each unique ID
        if unique_id_list:
            df = self._load_data()
            labels = []
            sequences = defaultdict(list)
            for unique_id in unique_id_list:
                unique_id_data = df[df[names.UNIQUE_ID] == unique_id]
                unique_id_sequences, unique_id_labels = self._generate_sequences(
                    unique_id_data, overrides=overrides
                )
                labels.extend(unique_id_labels)
                for name, sequence in unique_id_sequences.items():
                    sequences[name].extend(sequence)

            sequences = {
                name: np.array(sequence) for name, sequence in sequences.items()
            }
            return sequences, np.array(labels, dtype=np.float32)
        else:
            return self.sample_train_test_sequences(overrides=overrides)

    def create_dataloaders(self, return_class_weight=False):
        cpu_workers = (
            multiprocessing.cpu_count()
            if self.params.cpu_workers == -1
            else self.params.cpu_workers
        )

        train_sequences, train_targets, test_sequences, test_targets = (
            self.sample_train_test_sequences()
        )
        sequence_augmentations = (
            self.params.sequence_augmentations()
            if self.params.sequence_augmentations_config is not None
            else None
        )
        train_dataset = CustomImageDataset(
            train_sequences,
            train_targets,
            self.params.image_transforms(),
            sequence_augmentations=sequence_augmentations,
        )
        validation_dataset = CustomImageDataset(
            test_sequences, test_targets, self.params.image_transforms()
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=cpu_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=cpu_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        if return_class_weight:
            class_weights = []
            for i, num_classes in enumerate(self.params.classification_tasks.values()):
                target = train_targets[:, :, i]
                class_weights.extend(self._calculate_class_weights(target, num_classes))
            return train_dataloader, validation_dataloader, class_weights

        return train_dataloader, validation_dataloader

    def _calculate_class_weights(self, target, num_classes):
        if num_classes == 2:  # binary case, return weight for positive class
            return [np.sum(target == 0) / np.sum(target == 1)]
        else:  # multi-class case, return weight for each class
            return [1.0 for _ in range(num_classes)]
