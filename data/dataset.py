import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, VivitImageProcessor
from typing import Optional, Tuple, Dict, Sequence, List
from transformers.image_processing_utils import get_size_dict
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
import names


class ImageTransforms:
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        interpolation: int = Image.BICUBIC,
        normalize: bool = True,
    ):
        self.image_size = image_size
        self.interpolation = interpolation
        self.normalize = normalize
        # Fixed for all processors - based on dataset
        self.means = (0.35623488, 0.36024293, 0.33426452)
        self.stds = (0.20432226, 0.21891466, 0.22037318)

    def get_transforms(self):
        transform_list = [A.Resize(height=self.image_size[0], width=self.image_size[1])]
        if self.normalize:
            transform_list.append(A.Normalize(mean=self.means, std=self.stds))
        transform_list.append(ToTensorV2())

        return A.Compose(transform_list)

    @classmethod
    def from_pretrained(cls, config: str):
        try:
            processor = AutoImageProcessor.from_pretrained(config)
        except:
            try:
                processor = AutoProcessor.from_pretrained(config)
            except:
                processor = VivitImageProcessor.from_pretrained(config)

        processor_keys = processor.to_dict().keys()
        if "do_center_crop" in processor_keys and processor.do_center_crop:
            size = get_size_dict(processor.crop_size)
        else:
            size = get_size_dict(processor.size)

        size = list(get_size_dict(processor.size).values())[0]
        size = (size, size)

        if "do_normalize" in processor_keys:
            normalize = processor.do_normalize

        if "resample" in processor_keys:
            interpolation = processor.resample

        return cls(image_size=size, interpolation=interpolation, normalize=normalize)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        inputs: Dict[str, np.ndarray],
        targets: torch.Tensor,
        image_transform=None,
        sequence_augmentations=None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.image_transform = image_transform if image_transform else ToTensorV2()
        self.sequence_augmentations = sequence_augmentations
        self.fusion_names = list(inputs.keys())
        self.fusion_names.remove(names.IMAGES)
        self.fusion = len(self.fusion_names) > 0

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        label = self.targets[idx]
        image_paths = self.inputs[names.IMAGES][idx]
        images = [
            self.load_image(image_path) for image_path in image_paths
        ]  # returns list of numpy arrays

        if self.sequence_augmentations:
            images = self.apply_transform_to_sequence(images)
        images = torch.stack(images)

        if self.fusion:
            fusion_features = []
            for name in self.fusion_names:
                fusion_features.append(
                    torch.tensor(self.inputs[name][idx], dtype=torch.float32)
                )

            fusion_features = torch.stack(fusion_features, dim=1)
            return {names.IMAGES: images, names.FUSION_FEATURES: fusion_features}, label
        else:
            return {names.IMAGES: images}, label

    def load_image(self, img_path: str) -> np.ndarray:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_transform:
            image = self.image_transform(image=image)["image"]
        return image

    def apply_transform_to_sequence(self, images: Sequence[torch.Tensor]):
        numpy_images = [img.permute(1, 2, 0).numpy() for img in images]
        data = {"image": numpy_images[0]}
        for i, img in enumerate(numpy_images[1:], 1):
            data[f"image{i}"] = img

        # apply augmentations to all images
        transformed = self.sequence_augmentations(**data)
        transformed_images = [transformed["image"]] + [
            transformed.get(f"image{i}") for i in range(1, len(numpy_images))
        ]

        return transformed_images
