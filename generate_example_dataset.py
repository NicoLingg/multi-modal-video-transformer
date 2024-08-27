import os
import names
import random
import string
import shutil
import argparse
import pandas as pd
from PIL import Image


def generate_fake_data(num_samples, sequence_length):
    data = {
        names.UNIQUE_ID: [],
        names.TIME: [],
        names.IMAGE_PATH: [],
    }
    for input in names.ALL_FUSION_INPUTS:
        data[input] = []
    for task in names.ALL_TASKS:
        data[task] = []

    for i in range(num_samples):
        unique_id = "".join(random.choices(string.ascii_letters + string.digits, k=4))
        if i == 0:
            unique_id = "bFv8"  # Will generate one sequence with this ID and used as test ID, change in parameters.py if needed

        frames_dir = os.path.join(names.DATASET_DIR, f"{unique_id}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        for j in range(sequence_length):
            data[names.UNIQUE_ID].append(unique_id)
            data[names.TIME].append(f"{i * sequence_length + j}")

            image = Image.new(
                "RGB",
                (224, 224),
                color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            )
            image_name = f"{str(j).zfill(8)}.jpg"
            image_path = os.path.join(frames_dir, image_name)
            image.save(image_path)
            data[names.IMAGE_PATH].append(image_name)

            for input in names.ALL_FUSION_INPUTS:
                data[input].append(random.random())
            for task in names.ALL_REGRESSION_TASKS:
                data[task].append(random.random())
            for task, num_classes in zip(
                names.ALL_CLASSIFICATION_TASKS, names.NUM_CLASSES
            ):
                data[task].append(random.randint(0, num_classes - 1))

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(names.DATASET_DIR, names.DATA), index=False)

    print(f"Example dataset generated at {names.DATASET_DIR}")


# Example usage:
# 1. Run with default values:
#    python generate_example_dataset.py
#
# 2. Specify number of samples and sequence length:
#    python generate_example_dataset.py --num_samples 10 --sequence_length 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate fake data for example dataset"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=100, help="Length of each sequence"
    )
    args = parser.parse_args()

    num_samples = args.num_samples
    sequence_length = args.sequence_length

    assert num_samples >= 2, "num_samples should be at least 2"

    if os.path.exists(names.DATASET_DIR):
        shutil.rmtree(names.DATASET_DIR)
    os.makedirs(names.DATASET_DIR)

    generate_fake_data(num_samples, sequence_length)
