import pandas as pd
import numpy as np
from pathlib import Path


def load_artifact_data(file_path):
    data = pd.read_csv(
        file_path,
        dtype={"input_pattern": str, "target_pattern": str},
    )

    inputs = []
    targets = []

    for _, row in data.iterrows():
        input_vector = np.array(
            [int(value) for value in row["input_pattern"]],
            dtype=int
        )

        target_vector = np.array(
            [int(value) for value in row["target_pattern"]],
            dtype=int
        )

        inputs.append(input_vector)
        targets.append(target_vector)

    return np.array(inputs), np.array(targets)


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "artifact_pairs.csv"
    X, Y = load_artifact_data(dataset_path)

    print("Input patterns shape:", X.shape)
    print("Target patterns shape:", Y.shape)