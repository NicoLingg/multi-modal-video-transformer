# Multi-Modal Video Transformer For Trust Prediction

This framework is developed as part of the research presented in the paper **"Trust Prediction in Assistive Robotics using Multi-Modal Video Transformers"**. While the dataset used in the original research is not publicly available, this framework aims to serve as a valuable resource and inspiration for researchers and practitioners working on similar problems. The modular structure of the framework is designed to facilitate experiments with various models, tasks, and domains, enabling users to adapt it to their specific datasets and research questions.

# Framework Architecture
The following diagram illustrates the architecture of the multi-modal, multi-task Transformer framework:

![Framework Overview](./assets/network.png)

The framework consists of several key components:

1. **Vision Encoder**:
   - The framework leverages pre-trained vision models as a starting point, enabling users to load and fine-tune them for specific tasks.
   - It utilizes the Hugging Face library as an interface to load models as needed, making it easily extendable to support various vision models.

2. **Projector**:
   - The projector module is responsible for reducing the dimensionality of the feature representations obtained from the vision encoder.
   - It employs a multi-layer perceptron (MLP) with a single hidden layer to project the features to a lower-dimensional space.

3. **Fusion Module**:
   - The fusion module combines the projected visual features with additional input modalities.
   - It allows the framework to leverage multiple sources of information for enhanced learning and prediction.
   - The fusion is performed through element-wise addition of the projected visual features and the linearly transformed input modalities.

4. **Sequence Encoder**:
    - The sequence encoder processes the fused features temporally, capturing the sequential dependencies in the data.
   - It supports various architectures, including encoder-only Transformers, Long Short-Term Memory (LSTM) networks, and Multi-Layer Perceptrons (MLPs).
   - For Transformer-based models, the framework utilizes the Hugging Face interface for easy configuration and initialization of the Transformer architecture.
   - The sequence encoder can handle variable-length sequences and can be easily configured with different numbers of layers, attention heads, and hidden dimensions.

5. **Prediction Heads**:
   - The framework supports both classification and regression tasks through separate prediction heads.
   - The output of the sequence encoder is passed through a shared linear layer, followed by task-specific layers for each prediction head. 
   - Head configuration can be set as either separate for each task or shared across tasks.
   - The classification head outputs class probabilities, while the regression head produces continuous values.

6. **Weighted Multi-Task Loss**:
   - The framework employs a weighted multi-task loss function to jointly optimize the model across multiple tasks.
   - The loss function aggregates the individual task losses, allowing for the prioritization of certain tasks or balancing the contributions of different tasks during training.
   - The weights for each task can be adjusted based on the importance or difficulty of the task.


## Getting Started

### Prerequisites

Ensure you have Python 3.6 or later installed on your system.  Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/NicoLingg/multi-modal-video-transformer.git
cd trust-predict-multimodal-transformer
```

# Usage Guide

This framework allows for the flexible setup of experiments through the `ExperimentParams` class. Users can define various parameters for their experiments, including model configurations, task specifications, and input modalities, among others. The `ExperimentParams` class serves as the cornerstone for configuring and launching experiments.

## Setting Up Experiments

To set up your experiments, you will primarily interact with the `ExperimentParams` class. This class enables you to specify the details of your experiment, such as the model, tasks to be predicted, loss weights, and any specific configurations for fusion or sequence models.

### Example Experiments

For practical examples of how to use the `ExperimentParams` class to configure your experiments, please refer to the `example_experiments.py` file in this repository. This file contains predefined experiments that illustrate how to:

- Predict specific tasks from video data using different model architectures.
- Perform task and input ablations to understand the impact of various components.
- Utilize different sequence processing backbones and architectural configurations.

These examples are designed to provide a clear understanding of how to leverage the `ExperimentParams` class for a wide range of experimental setups. By examining these examples, you can gain insights into how to customize the framework for your specific research needs.

## Generating Example Data

To help users get started and understand the data format expected by the DataLoader, we provide a script named `generate_example_dataset.py`. This script generates random data in the format that the DataLoader consumes, serving illustrative purposes to give users an idea of how the system works with input data.

### How to Generate Example Data

To generate example data using the provided script, follow these steps:

1. Open a terminal and navigate to the project directory.

2. Run the script using Python. You can use it with default values or specify custom parameters:

   - To run with default values (10 samples, sequence length of 100):
     ```
     python generate_example_dataset.py
     ```

   - To specify the number of samples and sequence length:
     ```
     python generate_example_dataset.py --num_samples 20 --sequence_length 150
     ```

   Replace `20` and `150` with your desired values for the number of samples and sequence length, respectively.

3. The script will create a directory named `example_dataset` in your project folder, containing:
   - A CSV file with the generated data
   - Subdirectories for each unique ID, containing randomly generated image files

Note: Running this script will overwrite any existing data in the `example_dataset` directory.

### Implementing Your Own Data

If you wish to implement your own data, you have two options:

1. **Adapt your data to match the expected format**: Ensure your data conforms to the structure and format expected by the `CustomImageDataset` class in `dataset.py` and the `VideoSampler` class in `sampler.py`.
2. **Modify the data handling classes**: If adapting your data is not feasible or desirable, you may modify the `CustomImageDataset` class in `dataset.py` and the `VideoSampler` class in `sample.py` to accommodate your data's specific format and structure.


### Training Models

After setting up your experiments and preparing your data, you can train the models specified in your experiment configurations using the `train.py` script. This script handles the training process, including data loading, model initialization, and evaluation.


# Research Hyperparameters and Augmentations

For detailed information about the hyperparameters and data augmentations used in our research, as referenced in the paper "Trust Prediction in Assistive Robotics using Multi-Modal Video Transformers", please refer to our [Hyperparameters.md](Hyperparameters.md) file.

This file includes:
- Comprehensive model training hyperparameters
- Detailed data augmentation specifications
- Hardware specifications used for training


## Citing This Work

If you find this framework useful in your research, please consider citing our paper. 

<!-- Below is the BibTeX entry for our work: -->

<!-- ```bibtex 
@inproceedings{lingg2024trustprediction,
  title={Trust Prediction in Assistive Robotics using Multi-Modal Video Transformers},
  author={Lingg, Nico and Demiris, Yiannis},
  booktitle={International Conference on Social Robotics},
  pages={start-end},
  year={2024},
  organization={Springer}
}
``` -->


## License

Distributed under the Apache License 2.0 License. See `LICENSE` for more information.