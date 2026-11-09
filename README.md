# Neural Nets From Scratch

A hands-on, session-by-session journey into how neural networks actually work, built up from first principles. Instead of reaching for a deep learning framework, each session starts with plain Python and gradually introduces NumPy, so every number a neuron produces can be traced back to a few lines of code you wrote yourself.

The repo is a learning log: small Jupyter notebooks, one per session, each building on the last.

## What this repo teaches

- **The single neuron**: output = Σ(inputᵢ × weightᵢ) + bias, computed by hand.
- **Scaling from 3 inputs to 4 and beyond**: why the hand-written formula stops being practical.
- **A layer of neurons**: several neurons sharing the same inputs, each with its own weights and bias.
- **Refactoring with loops**: using `zip` to compute a whole layer without hard-coding every term.
- **Vectorizing with NumPy**: replacing nested loops with `np.dot`, which is the idea behind efficient neural network math.
- **Batches of inputs**: feeding many samples at once and why the weights matrix is transposed (`inputs · Wᵀ + biases`) so shapes line up.
- **Stacking layers**: feeding one layer's outputs into the next to form a (still linear) multi-layer network.

## Sessions

| Notebook | Topic | Key ideas |
|---|---|---|
| [session-1 (basic neurons impl)](<session-1(basic neurons impl).ipynb>) | Neurons in pure Python | Single neuron, extended inputs, layer of 3 neurons, loop-based layer |
| [session-2 (basic neurons with numpy)](<session-2(basic neurons with numpy).ipynb>) | Same neurons, now with NumPy | Recaps session 1, then `np.dot` for a neuron, batches of inputs with `weights.T`, and two chained layers |

### A taste of the code

```python
inputs  = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases  = [2, 3, 0.5]

layer_outputs = []
for neuron_weights, bias in zip(weights, biases):
    neuron_output = sum(i * w for i, w in zip(inputs, neuron_weights)) + bias
    layer_outputs.append(neuron_output)
```

Session 2 swaps the loops for NumPy and handles a whole batch and two layers in a few lines:

```python
layer1_outputs = np.dot(inputs, weights.T) + biases
layer2_outputs = np.dot(layer1_outputs, weights2.T) + biases2
```

## Getting started

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run jupyter lab   # or open the notebooks in your IDE with the .venv kernel
```

Dependencies (see [pyproject.toml](pyproject.toml)): `numpy` and `ipykernel`.

## Who it's for

Anyone who wants to understand what happens inside a neural network before using PyTorch or TensorFlow: students, engineers moving into ML, or anyone who learns best by building things themselves.
