# This file contains the default values for the pretrained processors & models from HF.
default_image_processors = {
    # Only defined for vision & video models
    "ResNet": "microsoft/resnet-50",
    "CLIP": "openai/clip-vit-base-patch32",
    "ViT": "google/vit-base-patch16-224",
    "ConvNext": "facebook/convnextv2-tiny-1k-224",
    "DeiT": "facebook/deit-tiny-patch16-224",
    "Focal": "microsoft/focalnet-tiny",
    "Swin": "microsoft/swin-tiny-patch4-window7-224",
}

default_model_configs = {
    "ResNet": "microsoft/resnet-50",
    "CLIP": "openai/clip-vit-base-patch32",
    "ViT": "google/vit-base-patch16-224",
    "ConvNext": "facebook/convnextv2-tiny-1k-224",
    # https://huggingface.co/docs/transformers/model_doc/convnext
    "DeiT": "facebook/deit-tiny-patch16-224",
    # https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/deit
    # facebook/deit-small-patch16-224, facebook/deit-base-patch16-224, facebook/deit-base-patch16-384
    "Focal": "microsoft/focalnet-tiny",
    # https://huggingface.co/docs/transformers/main/model_doc/focalnet
    "Swin": "microsoft/swin-tiny-patch4-window7-224",
    "GPT": "gpt2",
    "Mistral": "mistralai/Mistral-7B-v0.1",
    "Llama": "princeton-nlp/Sheared-LLaMA-1.3B",
}
