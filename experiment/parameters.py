# This file contains the ExperimentParams class which is used to store all parameters for the experiment
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, field, asdict
import dataclasses
from typing import List, Tuple, Dict, Any, Optional, NamedTuple, Union
from uuid import uuid4
import shutil
from enum import Enum
from models.pretrained_defaults import default_image_processors, default_model_configs
from data.dataset import ImageTransforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import names


class ImageProcessorConfig(NamedTuple):
    image_size: Tuple[int, int] = (224, 224)
    pretrained: bool = True
    family: str = "ViT"
    config: Union[str, dict] = "default"


class VisionModelConfig(NamedTuple):
    family: str = (
        "ViT"  # ['ResNet', 'ViT', 'CLIP', 'DeiT', 'Swin', 'Focal', 'ConvNext']
    )
    config: Union[str, dict] = "default"
    pretrained: bool = True
    train_weights: bool = False
    train_embedding: Optional[bool] = False
    num_blocks_trained: Optional[int] = 0
    pool: Optional[str] = "cls"  # 'avg', 'cls'. 'cls' only for some models


class SequenceModelConfig(NamedTuple):
    family: str = "Mistral"  # ['GPT', 'Mistral', 'Llama', 'Transformer', 'LSTM', 'MLP']
    config: Union[str, dict] = {
        "num_hidden_layers": 8,
        "intermediate_size": 1024,
        "hidden_size": 512,
        "num_attention_heads": 8,
        "attention_dropout": 0.0,
        "num_key_value_heads": 8,
        "max_position_embeddings": 200,
    }
    pretrained: bool = False
    train_weights: bool = True
    num_blocks_trained: Optional[int] = 8
    pool: Optional[str] = "last"  # 'avg',  'last'.


class FusionModelConfig(NamedTuple):
    fusion_type: str = "add"  # ['concat', 'add']
    fusion_features: List[str] = names.ALL_FUSION_INPUTS
    hidden_dim: int = 0
    output_dim: Optional[int] = None  # Only applicable for concat fusion
    dropout_p: Optional[float] = 0.0


class LRSchedulerConfig(NamedTuple):
    scheduler_type: str = (
        "CosineAnnealingLR"  # ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'StepRL', 'ExponentialLR']
    )
    T_max: Optional[int] = (
        25  # CosineAnnealingLR - Maximum number of iterations. if None, will be set as per batch annealing.
    )
    eta_min: Optional[float] = (
        0.00001  # CosineAnnealingLR, CosineAnnealingWarmRestarts -  Minimum learning rate.
    )
    warmup_steps: Optional[int] = 500  # Warmup steps for linear warmup
    step_size: Optional[int] = None  # StepLR
    gamma: Optional[float] = None  # StepLR, ReduceLROnPlateau, ExponentialLR
    patience: Optional[int] = None  # ReduceLROnPlateau
    last_epoch: Optional[int] = (
        -1
    )  # CosineAnnealingLR, CosineAnnealingWarmRestarts - The index of last epoch. Default: -1.
    T_0: Optional[int] = (
        None  # CosineAnnealingWarmRestarts - Number of iterations for the first restart.
    )
    T_mult: Optional[int] = (
        1  # CosineAnnealingWarmRestarts - A factor increases :math:`T_{i}` after a restart. Default: 1.
    )


@dataclass
class ExperimentParams:
    """Class to hold all parameters for the experiment"""

    # Metadata
    model_name: str = "model"
    model_description: str = "No description provided"

    # Set during post_init
    run_id: str = None
    run_date: str = str(pd.Timestamp.now())

    # Sampler
    sequence_length: int = 20
    target_length: int = 1
    downsample_factor: int = 2
    stride: int = 10
    test_ids: List[str] = field(default_factory=lambda: names.TEST_IDS)
    classification_tasks: Dict[str, int] = field(
        default_factory=lambda: names.CLASSIFICATION_TASKS_DICT
    )
    regression_tasks: List[str] = field(
        default_factory=lambda: names.ALL_REGRESSION_TASKS
    )
    cpu_workers: int = -1  # -1 signifies usage of all available CPU cores
    image_processor_config: ImageProcessorConfig = field(
        default_factory=lambda: ImageProcessorConfig()
    )
    enable_class_balancing: bool = True
    sequence_augmentations_config: Union[Dict[str, Dict[str, Any]], "str"] = "default"

    # Model: vision model -> projector -> sequence model -> head -> prediction
    vision_config: VisionModelConfig = field(
        default_factory=lambda: VisionModelConfig()
    )
    sequence_config: SequenceModelConfig = field(
        default_factory=lambda: SequenceModelConfig()
    )
    fusion_config: FusionModelConfig = field(
        default_factory=lambda: FusionModelConfig()
    )
    projector_hidden_dim: int = (
        256  # Setting to 0 will remove the hidden layer in projector
    )
    dropout_p: float = 0.1
    head_hidden_dim: int = 128  # Setting to 0 will remove the hidden layer in head
    split_prediction_tasks: bool = True
    seed: int = 452
    pretrained_weights_dir: Optional[str] = (
        None  # Directory to load pretrained weights from
    )
    sigma_reparam: bool = True

    # Training
    loss_weights: List[float] = field(
        default_factory=lambda: [1.0] * (len(names.ALL_TASKS))
    )
    lr: float = 0.0001  # AdamW only at the moment
    weight_decay: float = 0.01
    num_epochs: int = 1
    batch_size: int = 2
    lrs_config: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    enable_amp: bool = True

    def __post_init__(self):
        date, time = self.run_date.split(" ")
        date = date.replace("-", "")
        time = time.replace(":", "").split(".")[0]
        self.run_id = (
            f"{date}_{time}_{str(uuid4())[:8]}" if self.run_id is None else self.run_id
        )
        assert self.sequence_length > 0, "sequence_length must be an integer >= 1"
        assert self.target_length > 0, "target_length must be an integer >= 1"
        assert (
            self.target_length <= self.sequence_length
        ), "target_length must be less than or equal to sequence_length"
        assert self.downsample_factor > 0, "downsample_factor must be an integer >= 1"
        assert self.stride > 0, "stride must be an integer >= 1"
        assert len(self.regression_tasks) + len(self.classification_tasks) == len(
            self.loss_weights
        ), "Incorrect number of loss weights"
        if self.sequence_augmentations_config == "default":
            self.sequence_augmentations_config = self.default_sequence_augmentations
        if (
            self.image_processor_config.pretrained
            and self.image_processor_config.config == "default"
        ):
            self.image_processor_config = self.image_processor_config._replace(
                config=default_image_processors[self.image_processor_config.family]
            )
        if self.image_processor_config is not None:
            if (
                self.image_processor_config.config == "default"
                and self.image_processor_config.pretrained
            ):
                self.image_processor_config = self.image_processor_config._replace(
                    config=default_image_processors[self.image_processor_config.family]
                )
        if self.fusion_config:
            assert self.fusion_config.fusion_type in [
                "concat",
                "add",
            ], "Fusion type must be 'concat' or 'add'"

        if self.vision_config.config == "default" and self.vision_config.pretrained:
            self.vision_config = self.vision_config._replace(
                config=default_model_configs[self.vision_config.family]
            )
        if self.sequence_config.config == "default" and self.sequence_config.pretrained:
            self.sequence_config = self.sequence_config._replace(
                config=default_model_configs[self.sequence_config.family]
            )
        if self.sequence_config.family in ["LSTM"]:
            assert self.target_length == 1, "LSTM only supports target_length = 1"
        if self.sequence_config.family == "MLP":
            self.sequence_config.config["sequence_length"] = self.sequence_length

    def as_dict(self):
        params_dict = asdict(self)
        for config in [
            "vision_config",
            "sequence_config",
            "fusion_config",
            "lrs_config",
            "image_processor_config",
        ]:
            if params_dict[config] is not None:
                params_dict[config] = params_dict[config]._asdict()

        return params_dict

    def save(self):
        params_dir = os.path.join(names.MODELS_DIR, f"{self.model_name}_{self.run_id}")
        os.makedirs(params_dir, exist_ok=True)
        params_file = os.path.join(params_dir, names.EXP_PARAMS)
        params_dict = self.as_dict()

        with open(params_file, "w") as f:
            json.dump(params_dict, f, indent=4)

    def clean_up(self, model_only=False):
        params_dir = os.path.join(names.MODELS_DIR, f"{self.model_name}_{self.run_id}")

        if model_only:
            for filename in os.listdir(params_dir):
                if filename.endswith(".pt"):
                    os.remove(os.path.join(params_dir, filename))
        else:
            shutil.rmtree(params_dir)

    def image_transforms(self):
        if self.image_processor_config.pretrained:
            image_transform = ImageTransforms.from_pretrained(
                self.image_processor_config.config
            )
        else:
            image_transform = ImageTransforms(
                image_size=self.image_processor_config.image_size
            )
        return image_transform.get_transforms()

    def update_param(self, param, value, save=True):
        if hasattr(self, param):
            setattr(self, param, value)
        else:
            raise ValueError(f"Attribute {param} does not exist in ExperimentParams.")
        if save:
            self.save()

    def replace(self, **kwargs):
        new_params = dataclasses.replace(
            self, run_id=None, run_date=str(pd.Timestamp.now()), **kwargs
        )
        return new_params

    def sequence_augmentations(self, with_keypoints=False):
        if with_keypoints:  # Can be used to transform fusion inputs with images
            sequence_augmentations = A.Compose(
                [
                    self.sequence_augmentations_mapping[name](**params)
                    for name, params in self.sequence_augmentations_config.items()
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
                additional_targets={
                    **{
                        "image" + str(i): "image"
                        for i in range(1, self.sequence_length)
                    },
                    **{
                        "keypoints" + str(i): "keypoints"
                        for i in range(1, self.sequence_length)
                    },
                },
            )
        else:
            sequence_augmentations = A.Compose(
                [
                    self.sequence_augmentations_mapping[name](**params)
                    for name, params in self.sequence_augmentations_config.items()
                ],
                additional_targets={
                    **{
                        "image" + str(i): "image"
                        for i in range(1, self.sequence_length)
                    }
                },
            )

        return sequence_augmentations

    @property
    def sequence_augmentations_mapping(self):
        return {
            "RandomPerspective": A.Perspective,
            "RandomBrightnessContrast": A.RandomBrightnessContrast,
            "RandomSharpen": A.Sharpen,
            "RandomAffine": A.Affine,
            "Blur": A.Blur,
            "CropAndPad": A.CropAndPad,
            "Emboss": A.Emboss,
            "HueSaturationValue": A.HueSaturationValue,
            "ToTensorV2": ToTensorV2,
        }

    @property
    def default_sequence_augmentations(self):
        return {
            "RandomPerspective": {"scale": [0.05, 0.1], "p": 0.2},
            "RandomBrightnessContrast": {"brightness_limit": [-0.2, 0.2], "p": 0.2},
            "RandomSharpen": {"alpha": [0.2, 0.5], "lightness": [0.5, 1.0], "p": 0.2},
            "RandomAffine": {
                "scale": [0.95, 1.05],
                "translate_percent": [-0.1, 0.1],
                "rotate": 0,
                "shear": 0,
                "p": 0.2,
            },
            "Blur": {"blur_limit": 4, "p": 0.2},
            "CropAndPad": {"percent": 0.05, "p": 0.2},
            "Emboss": {"alpha": [0.2, 0.5], "strength": [0.2, 0.7], "p": 0.2},
            "HueSaturationValue": {
                "hue_shift_limit": 20,
                "sat_shift_limit": 30,
                "val_shift_limit": 20,
                "p": 0.2,
            },
            "ToTensorV2": {"p": 1},
        }

    @classmethod
    def load_from_subfolder(cls, sub_folder):
        params_dir = os.path.join(names.MODELS_DIR, sub_folder)
        params_file = os.path.join(params_dir, names.EXP_PARAMS)

        with open(params_file, "r") as f:
            params_dict = json.load(f)

        # Convert dict back to ModelConfig NamedTuple
        params_dict["vision_config"] = (
            VisionModelConfig(**params_dict["vision_config"])
            if params_dict["vision_config"] is not None
            else None
        )
        params_dict["sequence_config"] = (
            SequenceModelConfig(**params_dict["sequence_config"])
            if params_dict["sequence_config"] is not None
            else None
        )
        params_dict["fusion_config"] = (
            FusionModelConfig(**params_dict["fusion_config"])
            if params_dict["fusion_config"] is not None
            else None
        )
        params_dict["image_processor_config"] = (
            ImageProcessorConfig(**params_dict["image_processor_config"])
            if params_dict["image_processor_config"] is not None
            else None
        )
        params_dict["lrs_config"] = (
            None
            if params_dict["lrs_config"] is None
            else LRSchedulerConfig(**params_dict["lrs_config"])
        )

        return cls(**params_dict)

    @classmethod
    def clean_up_subfolder(cls, sub_folder, model_only=False):
        params_dir = os.path.join(names.MODELS_DIR, sub_folder)

        if model_only:
            for filename in os.listdir(params_dir):
                if filename.endswith(".pt"):
                    os.remove(os.path.join(params_dir, filename))
        else:
            shutil.rmtree(params_dir)
