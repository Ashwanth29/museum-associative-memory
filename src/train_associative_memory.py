import numpy as np
from prepare_data import load_artifact_data


def bipolar_encode(pattern):
    return np.where(pattern == 0, -1, 1)


def train_associative_memory(inputs, targets):
    input_size = inputs.shape[1]
    output_size = targets.shape[1]

    weights = np.zeros((output_size, input_size))

    for x, y in zip(inputs, targets):
        x_bipolar = bipolar_encode(x)
        y_bipolar = bipolar_encode(y)

        weights += np.outer(y_bipolar, x_bipolar)

    return weights


if __name__ == "__main__":
    X, Y = load_artifact_data("dataset/artifact_pairs.csv")

    weights = train_associative_memory(X, Y)

    print("Training completed.")
    print("Weight matrix shape:", weights.shape)