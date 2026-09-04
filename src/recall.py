import numpy as np
import matplotlib.pyplot as plt
from prepare_data import load_artifact_data
from train_associative_memory import (
    train_associative_memory,
    bipolar_encode
)


def recall_pattern(input_pattern, weights):
    bipolar_input = bipolar_encode(input_pattern)

    activation = np.dot(weights, bipolar_input)

    recalled = np.where(activation >= 0, 1, 0)

    return recalled
def calculate_accuracy(actual, predicted):
    return np.mean(actual == predicted)

def display_pattern(pattern, title):
    image = pattern.reshape(5, 5)

    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    X, Y = load_artifact_data("dataset/artifact_pairs.csv")

    weights = train_associative_memory(X, Y)

    test_input = X[0]

    recalled_output = recall_pattern(test_input, weights)

    accuracy = calculate_accuracy(Y[0], recalled_output)

    print("\nRecall accuracy:", accuracy * 100, "%")

    print("Input pattern:")
    print(test_input)

    print("\nRecalled restoration:")
    print(recalled_output)

    print("\nExpected restoration:")
    print(Y[0])

    display_pattern(test_input, "Damaged Artifact")
    display_pattern(recalled_output, "Recalled Restoration")
    display_pattern(Y[0], "Expected Restoration")