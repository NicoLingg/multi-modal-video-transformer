# Main file for training the models and running the experiments.

from experiment.runner import run_experiment
from example_experiments import exp1_models


def main():
    for exp in exp1_models:
        run_experiment(exp)


if __name__ == "__main__":
    main()
