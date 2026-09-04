# Training Algorithms for Pattern Association

## Museum Artifact Restoration Using Associative Memory

## 1. Introduction

This project demonstrates pattern association using an
associative-memory neural network.

A museum has digitized damaged artifact symbols and their
corresponding restoration patterns. The objective is to learn
the relationship between the damaged input symbols and their
restored output patterns.

## 2. Objective

The objectives are:

- Prepare damaged/restored pattern pairs.
- Represent symbols as binary vectors.
- Train an associative-memory network.
- Learn the input-output relationship.
- Recall a restoration from a damaged pattern.
- Evaluate the recalled pattern.

## 3. Methodology

The project uses hetero-associative memory with Hebbian learning.

The training process is:

Damaged Pattern
→ Binary Encoding
→ Hebbian Learning
→ Weight Matrix
→ Pattern Recall
→ Restored Pattern

## 4. Algorithm

The weight matrix is calculated using:

W = Σ yxᵀ

During recall:

y = sign(Wx)

## 5. Dataset

The dataset contains pairs of:

- Damaged artifact symbols
- Corresponding restored symbols

Each symbol is represented using a 5 × 5 binary pattern.

## 6. Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## 7. Project Structure

museum-associative-memory/

dataset/
src/
notebooks/
results/
screenshots/

## 8. Results

The trained associative-memory network recalls the restoration
pattern associated with the supplied damaged artifact.

## 9. Conclusion

The project demonstrates how associative-memory networks learn
relationships between input and output patterns and use the
learned association for recall.