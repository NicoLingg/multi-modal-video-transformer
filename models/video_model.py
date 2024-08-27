# This file contains the VideoModel and components.
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoModel,
    CLIPVisionModel,
    SwinConfig,
    ConvNextConfig,
    FocalNetConfig,
    DeiTConfig,
    MistralConfig,
    ResNetConfig,
    LlamaConfig,
    GPT2Config,
    ViTConfig,
    CLIPVisionConfig,
)
from torch.utils.data import DataLoader
import os

from typing import List, Union, Optional
from tqdm import tqdm
import multiprocessing
import names
from models.modules import TransformerModel, LSTMModel, MLPModel, create_layers
from models import sigma_reparam
from data.dataset import CustomImageDataset
from experiment.parameters import (
    ExperimentParams,
    VisionModelConfig,
    SequenceModelConfig,
    FusionModelConfig,
)


class BaseModel(nn.Module):
    def __init__(
        self, config: Union[VisionModelConfig, SequenceModelConfig, FusionModelConfig]
    ):
        super().__init__()
        self.config = config

    def save(self, path):
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), path)

    def get_output_dim(self):
        raise NotImplementedError


class FusionModel(BaseModel):
    def __init__(
        self, config: FusionModelConfig, sequence_embedding_dim: Optional[int] = None
    ):
        super().__init__(config)
        input_dim = len(config.fusion_features)
        self.output_dim = (
            config.output_dim
            if config.fusion_type == "concat"
            else sequence_embedding_dim
        )
        if config.fusion_type == "concat" and config.output_dim == input_dim:
            layers = [nn.Identity()]
        else:
            layers = create_layers(
                input_dim,
                sizes=[config.hidden_dim, self.output_dim],
                dropout_p=config.dropout_p,
            )
        self.model = nn.Sequential(*layers)

    def forward(self, fusion_features, projector_outputs):
        fusion_outputs = self.model(fusion_features)
        if self.config.fusion_type == "concat":
            return torch.cat([projector_outputs, fusion_outputs], dim=-1)
        else:
            return projector_outputs + fusion_outputs

    def get_output_dim(self):
        return self.output_dim

    @property
    def fusion_types(self):
        return self.config.fusion_type


class VisionModel(BaseModel):
    def __init__(self, config: VisionModelConfig):
        super().__init__(config)
        supported_models = [
            "ResNet",
            "CLIP",
            "ViT",
            "DeiT",
            "Swin",
            "Focal",
            "ConvNext",
        ]
        assert (
            config.family in supported_models
        ), f"VisionModel family must be one of {supported_models}"
        if config.pretrained:
            if config.family == "CLIP":
                # AutoModel for CLIP defaults to multimodel CLIPModel
                self.model = CLIPVisionModel.from_pretrained(config.config).vision_model
            else:
                self.model = AutoModel.from_pretrained(config.config)
        else:
            model_config = self.family_mapping[config.family](**config.config)
            self.model = AutoModel.from_config(model_config)

        for param in self.model.parameters():
            param.requires_grad = False

        if config.train_weights:
            if config.train_embedding:
                # Train conv net embedding
                embeddings = (
                    self.model.embedder
                    if config.family == "ResNet"
                    else self.model.embeddings
                )
                for param in embeddings.parameters():
                    param.requires_grad = True

            if config.num_blocks_trained > 0:
                if config.family in ["ResNet", "Focal", "ConvNext"]:
                    # Treat stages in resnet like layers in transformers
                    layers = self.model.encoder.stages
                elif config.family in ["ViT", "DeiT"]:
                    layers = self.model.encoder.layer
                else:
                    # CLIP or Swin
                    self.model.encoder.layers
                num_layers = len(layers)
                for n in range(num_layers - config.num_blocks_trained, num_layers):
                    for param in layers[n].parameters():
                        param.requires_grad = True

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch * num_frames, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch * num_frames, hidden_size).
        """
        # x.shape = batch * num_channels, height, width
        outputs = self.model(x, return_dict=True)
        if self.config.family in ["ResNet", "Swin", "Focal", "ConvNext"]:
            # Pooler output is result of adaptive pooling over spatial dimensions
            return outputs["pooler_output"]
        else:
            # CLIP or ViT. Pooler output is results of linear layer over cls token
            # Ignore pooler output, rather control this with projection
            if self.config.pool == "avg":
                return outputs["last_hidden_state"].mean(dim=1)
            else:
                # self.config.pool == 'cls'
                return outputs["last_hidden_state"][:, 0]

    def get_output_dim(self):
        if self.config.family in ["Focal", "ResNet", "ConvNext"]:
            return self.model.config.hidden_sizes[-1]
        return self.model.config.hidden_size

    @property
    def family_mapping(self):
        return {
            "ResNet": ResNetConfig,
            "ViT": ViTConfig,
            "CLIP": CLIPVisionConfig,
            "DeiT": DeiTConfig,
            "Swin": SwinConfig,
            "Focal": FocalNetConfig,
            "ConvNext": ConvNextConfig,
        }


class SequenceModel(BaseModel):
    def __init__(self, config: SequenceModelConfig):
        super().__init__(config)
        supported_models = ["GPT", "Llama", "Transformer", "Mistral", "LSTM", "MLP"]
        assert (
            config.family in supported_models
        ), f"Only {supported_models} SequenceModel family supported"
        if config.family == "Transformer":
            self.model = (
                TransformerModel.from_pretrained(config.config)
                if config.pretrained
                else TransformerModel(**config.config)
            )
        elif config.family == "LSTM":
            self.model = (
                LSTMModel.from_pretrained(config.config)
                if config.pretrained
                else LSTMModel(**config.config)
            )
            assert config.pool == "last", "LSTM only supports last pooling"
        elif config.family == "MLP":  # MLP over time
            self.model = MLPModel(**config.config)
        else:
            if config.pretrained:
                self.model = AutoModel.from_pretrained(config.config)
            else:
                model_config = self.family_mapping[config.family](**config.config)
                self.model = AutoModel.from_config(model_config)

        for param in self.model.parameters():
            param.requires_grad = False

        if config.train_weights:
            if config.family == "LSTM":
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                if config.num_blocks_trained > 0:
                    if config.family == "GPT":
                        layers = self.model.h
                    else:
                        layers = self.model.layers
                    num_layers = len(layers)
                    for n in range(num_layers - config.num_blocks_trained, num_layers):
                        for param in layers[n].parameters():
                            param.requires_grad = True

    def forward(self, x, target_length):
        """Performs a forward pass through the network. Intented to follow a ProjectorModel.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch, num_frames, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch, hidden_size) corresponding
                to the last hidden state of the sequence model.
        """
        # pooling is always last for  LSTM
        if self.config.family == "LSTM":
            return self.model(x)
        else:
            if self.config.family in ["Transformer", "MLP"]:
                output = self.model(x)
            else:
                x = self.model(inputs_embeds=x)
                output = x.last_hidden_state

            if self.config.pool == "avg":
                # Average pooling over time
                return output.mean(dim=1, keepdim=True)
            else:
                # self.config.pool == 'last'
                return output[:, -target_length:]

    def get_output_dim(self):
        if self.config.family == "GPT":
            return self.model.config.n_embd
        elif self.config.family in ["Transformer", "MLP", "LSTM"]:
            return self.model.hidden_size
        else:
            return self.model.config.hidden_size

    @property
    def family_mapping(self):
        return {
            "GPT": GPT2Config,
            "Llama": LlamaConfig,
            "Transformer": TransformerModel,
            "Mistral": MistralConfig,
            "LSTM": LSTMModel,
        }


class VideoModel(nn.Module):
    def __init__(self, params: ExperimentParams):
        super(VideoModel, self).__init__()

        self.model_dir = os.path.join(
            names.MODELS_DIR, f"{params.model_name}_{params.run_id}"
        )
        self.target_length = params.target_length

        # Load vision, fusion, and sequence models
        self.vision_model = VisionModel(params.vision_config)
        vision_output_dim = self.vision_model.get_output_dim()

        self.sequence_model = SequenceModel(params.sequence_config)
        sequence_embedding_dim = self.sequence_model.get_output_dim()

        if params.fusion_config:
            self.fusion_model = FusionModel(
                params.fusion_config, sequence_embedding_dim=sequence_embedding_dim
            )
            fusion_output_dim = self.fusion_model.get_output_dim()
            projector_output_dim = (
                sequence_embedding_dim - fusion_output_dim
                if params.fusion_config.fusion_type == "concat"
                else sequence_embedding_dim
            )
        else:
            self.fusion_model = None
            projector_output_dim = sequence_embedding_dim

        # Projector
        self.projector = nn.Sequential(
            *create_layers(
                input_dim=vision_output_dim,
                sizes=[params.projector_hidden_dim, projector_output_dim],
                dropout_p=params.dropout_p,
            )
        )

        # Common head
        self.head = nn.Sequential(
            *create_layers(
                input_dim=sequence_embedding_dim,
                sizes=[params.head_hidden_dim],
                dropout_p=params.dropout_p,
            )
        )

        head_output_dim = (
            params.head_hidden_dim
            if params.head_hidden_dim > 0
            else sequence_embedding_dim
        )

        # Fully connected layers with bias for different tasks
        if params.split_prediction_tasks:
            predictions = []
            for num_classes in params.classification_tasks.values():
                num_classes = 1 if num_classes == 2 else num_classes
                predictions.append(nn.Linear(head_output_dim, num_classes, bias=True))
            for _ in range(len(params.regression_tasks)):
                predictions.append(nn.Linear(head_output_dim, 1, bias=True))
            self.prediction = nn.ModuleList(predictions)
        else:
            class_outputs = [
                1 if n == 2 else n for n in params.classification_tasks.values()
            ]
            total_out = sum(class_outputs) + len(params.regression_tasks)
            self.prediction = nn.ModuleList(
                [nn.Linear(head_output_dim, total_out, bias=True)]
            )

        if params.sigma_reparam:
            self.sequence_model = sigma_reparam.convert_layers(self.sequence_model)
            self.projector = sigma_reparam.convert_layers(self.projector)

        # self.load_state(version='latest', pretrained_weights_dir=params.pretrained_weights_dir)

    def forward(self, images, fusion_features=None, target_length=None):
        target_length = target_length if target_length else self.target_length
        batch_size, sample_length, C, H, W = images.size()
        pixel_values = images.view(batch_size * sample_length, C, H, W)
        vision_outputs = self.vision_model(pixel_values).view(
            batch_size, sample_length, -1
        )
        projector_outputs = self.projector(vision_outputs)

        if self.fusion_model:
            projector_outputs = self.fusion_model(
                fusion_features, projector_outputs=projector_outputs
            )

        sequence_outputs = self.sequence_model(
            projector_outputs, target_length=target_length
        )

        # Outputs for different tasks
        head_outputs = self.head(sequence_outputs)
        prediction_outputs = torch.cat(
            [task(head_outputs) for task in self.prediction], dim=-1
        )

        return prediction_outputs

    def save(self, version=names.BEST_MODEL):
        torch.save(self.state_dict(), f"{self.model_dir}/{version}")
        if self.vision_model.config.train_weights:
            vision_model_dir = os.path.join(self.model_dir, names.VISION_MODEL_DIR)
            os.makedirs(vision_model_dir, exist_ok=True)
            self.vision_model.save(f"{vision_model_dir}/{version}")

        if self.sequence_model.config.train_weights:
            sequence_model_dir = os.path.join(self.model_dir, names.SEQUENCE_MODEL_DIR)
            os.makedirs(sequence_model_dir, exist_ok=True)
            self.sequence_model.save(f"{sequence_model_dir}/{version}")

    def load_state(
        self, version=names.BEST_MODEL, pretrained_weights_dir: Optional[str] = None
    ):
        weights_dir = (
            f"{names.MODELS_DIR}/{pretrained_weights_dir}"
            if pretrained_weights_dir
            else self.model_dir
        )
        model_file = os.path.join(weights_dir, version)

        if os.path.exists(model_file):
            print(f"Loading model from {model_file}")
            print(f"This model will be saved to {self.model_dir}")
            self.load_state_dict(
                torch.load(model_file, map_location=torch.device("cpu"))
            )
        else:
            print(f"No saved model found at {model_file}.")

    def predict(self, X, y, save_predictions_as=""):
        # Set device
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using {device} device")
        self.to(device)

        # Create a CustomImageDataset and DataLoader
        params = ExperimentParams.load_from_subfolder(self.model_dir.split("/")[-1])
        dataset = CustomImageDataset(X, y, image_transform=params.image_transforms())
        cpu_workers = (
            multiprocessing.cpu_count()
            if params.cpu_workers == -1
            else params.cpu_workers
        )
        dataloader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=cpu_workers,
        )

        # Perform inference
        results = []
        self.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Predicting"):
                images = inputs[names.IMAGES].to(device)
                fusion_features = (
                    inputs[names.FUSION_FEATURES].to(device)
                    if names.FUSION_FEATURES in inputs
                    else None
                )
                outputs = self(
                    images=images, fusion_features=fusion_features, target_length=1
                )
                results.append(outputs)

        results = torch.cat(results, dim=0)

        if save_predictions_as:
            results_dir = os.path.join(self.model_dir, names.RESULTS_DIR)
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, f"{save_predictions_as}.pt")
            results = {
                "inputs": {
                    k: v.to("cpu") if torch.is_tensor(v) else v for k, v in X.items()
                },
                "predictions": results.to("cpu"),
                "targets": torch.tensor(y).to("cpu"),
                "classification_tasks": params.classification_tasks,
                "regression_tasks": params.regression_tasks,
            }
            torch.save(results, results_path)

        return results
