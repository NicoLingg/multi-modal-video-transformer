# This file contains example experiments

from experiment.parameters import (
    ExperimentParams,
    FusionModelConfig,
    SequenceModelConfig,
)
import names

loss_weights_dict = {
    names.CT1: 0.4,
    names.CT2: 1.0,
    names.RT1: 10,
    names.RT2: 10,
}

# Example Experiment 1: Task and input ablations
model_arch = "Frozen ViT & Transformer for sequence processing. "

exp1_m01 = ExperimentParams(
    model_name="Exp1_M01_Video_T1",
    model_description="Predicting task 1 from video data. " + model_arch,
    classification_tasks={names.CT1: names.NT1},
    regression_tasks=[],
    loss_weights=[loss_weights_dict[names.CT1]],
    fusion_config=None,
)

exp1_m02 = exp1_m01.replace(
    model_name="Exp1_M02_Video_All_Classification_Tasks",
    model_description="Predicting both tasks from video data. " + model_arch,
    classification_tasks=names.CLASSIFICATION_TASKS_DICT,
    loss_weights=[loss_weights_dict[t] for t in names.ALL_CLASSIFICATION_TASKS],
)

exp1_m03 = exp1_m02.replace(
    model_name="Exp1_M03_Video_All_tasks",
    model_description="Predicting trust, performance & wheelchair from video data. "
    + model_arch,
    regression_tasks=names.ALL_REGRESSION_TASKS,
    loss_weights=[loss_weights_dict[t] for t in names.ALL_TASKS],
)

exp1_m04 = exp1_m03.replace(
    model_name="Exp1_M04_Video_All_tasks_Fusion_inputs",
    model_description="Predicting all tasks from video data with fusion. " + model_arch,
    fusion_config=FusionModelConfig(fusion_features=names.ALL_FUSION_INPUTS),
)

exp1_models = [exp1_m01, exp1_m02, exp1_m03, exp1_m04]

# Example Experiment 2: Sequence backbones and architecural ablations
model_data = "Predicting all tasks with all inputs. "

exp2_m01 = ExperimentParams(
    model_name="Exp2_M01_LSTM",
    model_description=model_data
    + "Frozen pretrained ViT & LSTM for sequence processing",
    batch_size=12,
    sequence_config=SequenceModelConfig(
        family="LSTM", config={"num_layers": 8, "hidden_size": 256, "dropout_p": 0.0}
    ),
)

exp2_m02 = exp2_m01.replace(
    model_name="Exp2_M02_MLP",
    model_description=model_data
    + "Frozen pretrained ViT & MLP for sequence processing",
    sequence_config=SequenceModelConfig(
        family="MLP", config={"num_layers": 8, "hidden_size": 256}
    ),
)

exp2_m03 = exp2_m01.replace(
    model_name="Exp2_M03_Avg_pool",
    model_description=model_data
    + "Frozen pretrained ViT & Avg pooling for sequence processing",
    sequence_config=SequenceModelConfig(
        family="MLP",
        config={"num_layers": 0, "hidden_size": 256},
        num_blocks_trained=0,
        train_weights=False,
        pool="avg",
    ),
)

exp2_m04 = exp2_m01.replace(
    model_name="T2_M04_Last_image",
    model_description=model_data
    + "Frozen pretrained ViT & No sequential processing, only lasy image",
    sequence_length=1,  # Only use the last image
    sequence_config=SequenceModelConfig(
        family="MLP",
        config={"num_layers": 0, "hidden_size": 256},
        num_blocks_trained=0,
        train_weights=False,
        pool="last",
    ),
)

exp2_models = [exp2_m01, exp2_m02, exp2_m03, exp2_m04]
