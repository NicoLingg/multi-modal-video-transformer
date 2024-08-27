# This file contains names used in the experiment pipeline.

# CLASSIFICATION TASKS
CT1 = "classification_task1"
CT2 = "classification_task2"
ALL_CLASSIFICATION_TASKS = [CT1, CT2]

# NUMBER OF CLASSES
NT1 = 3
NT2 = 2
NUM_CLASSES = [NT1, NT2]
CLASSIFICATION_TASKS_DICT = dict(zip(ALL_CLASSIFICATION_TASKS, NUM_CLASSES))


# REGRESSION TASKS
RT1 = "regression_task1"
RT2 = "regression_task2"
ALL_REGRESSION_TASKS = [RT1, RT2]

ALL_TASKS = ALL_CLASSIFICATION_TASKS + ALL_REGRESSION_TASKS

# IMAGE INPUTS
IMAGE_PATH = "image_path"

# FUSION INPUTS
FI1 = "fusion_input1"
FI2 = "fusion_input2"
ALL_FUSION_INPUTS = [FI1, FI2]

# MUST BE SPECIFIED AFTER DATASET GENERATION, CURRENLTY HARDCODED FOR TESTING IN generate_example_dataset.py
TEST_IDS = ["bFv8"]

# OTHER COLUMNS
TIME = "timestamp"
UNIQUE_ID = "id"

# DATALOADER NAMES
IMAGES = "images"
FUSION_FEATURES = "fusion_features"

# FOLDER NAMES
MODELS_DIR = "model_checkpoints"
SEQUENCE_MODEL_DIR = "sequence_model"
VISION_MODEL_DIR = "vision_model"
DATASET_DIR = "example_dataset"
FRAMES_DIR = "frames"
RESULTS_DIR = "results"

# FILE NAMES
DATA = "example_dataset.csv"
EXP_PARAMS = "exp_params.json"
METRICS = "metrics.json"
REPORT = "report.json"
BEST_MODEL = "best_model.pt"
LAST_MODEL = "last_model.pt"
ANIMATION = "animation.mp4"
