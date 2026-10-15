"""Build self-contained teaching notebooks with nbformat; execute separately."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells, filename):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata.language_info = {"name": "python", "version": "3.13"}
    nbf.validate(nb)
    nbf.write(nb, ROOT / "Lab01" / filename)


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


SETUP = '''
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({"figure.figsize": (9, 4), "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "animation.embed_limit": 25})
BLUE, ORANGE = "#2458a6", "#be5514"
print(f"Python {sys.version.split()[0]} | NumPy {np.__version__} | Matplotlib {matplotlib.__version__}")
'''

notebook([
md('''# Neural Networks · First laboratory
### Python, arrays, and the meaning of a weight

2 October 2026 · Cristian Simionescu · English student edition

## Goal
Represent a small image as numbers, predict array shapes, compute a batch of weighted scores, and explain why performance on unseen examples matters.

**Core route:** sections 1–7, then the exercises and exit ticket. The matrix animation is a short optional demonstration. Training is in a separate optional notebook.

This supplements the published 2026 Lab 1 and Course 1. All example data below are synthetic teaching data. The detector weights are hand-designed, not learned.

## Setup
Run cells in order with Shift+Enter. The only numerical dependencies are NumPy and Matplotlib. See `../SETUP.md` for installation. No network, GPU, or dataset download is needed after setup.'''),
code(SETUP),
md('''## Steps
### 1. Python warm-up: predict before running
What will `values * 2` do? What changes when `values` is an array?

A list is a general container. A numerical array has a shape and a shared dtype. Type annotations document intent; they do not automatically enforce types at runtime.'''),
code('''values = [1, 2, 3]
array = np.array(values, dtype=float)
print("List * 2:", values * 2)
print("Array * 2:", array * 2)
print("Positive entries:", np.array([-2, 0, 3, 5])[np.array([-2, 0, 3, 5]) > 0])

def weighted_sum(values, weights):
    if len(values) != len(weights):
        raise ValueError("One weight is required for each input")
    return sum(value * weight for value, weight in zip(values, weights))

print("Python weighted sum:", weighted_sum([2, 3], [0.5, -1]))
assert weighted_sum([2, 3], [0.5, -1]) == -2'''),
md('''### 2. A picture is an array
These are two 5×5 grayscale pictures. A value of 1 means bright and 0 means dark. Flattening preserves the values and changes how we address them.

**Predict:** How many inputs does one fully connected neuron need for one picture? What shape represents four pictures?'''),
code('''vertical = np.zeros((5, 5), dtype=float)
vertical[:, 2] = 1
horizontal = np.zeros((5, 5), dtype=float)
horizontal[2, :] = 1
shifted_vertical = np.roll(vertical, 1, axis=1)
blank = np.zeros((5, 5), dtype=float)
images = np.stack([vertical, horizontal, shifted_vertical, blank])
names = ["Vertical", "Horizontal", "Shifted vertical", "Blank"]
X = images.reshape(len(images), -1)
print("Images:", images.shape, "Batch of feature vectors:", X.shape)
assert X.shape == (4, 25)
assert np.array_equal(X[0].reshape(5, 5), vertical)

fig, axes = plt.subplots(1, 4, figsize=(11, 3), layout="constrained")
for ax, name, picture in zip(axes, names, images):
    ax.imshow(picture, cmap="gray", vmin=0, vmax=1)
    ax.set(title=name, xlabel="Pixel column", ylabel="Pixel row", xticks=[0, 2, 4], yticks=[0, 2, 4])
plt.show()'''),
md('''**Try:** change the central column to a diagonal using `np.eye(5)`. Is its shape different? Is its information different?

### 3. A weight says what to reward and what to penalize
The vertical detector rewards the centre column and penalizes the other pixels. A dot product multiplies each input by its weight, then adds the contributions.

For a single image, `score = x @ w + b`. These scores are arbitrary real values, not probabilities.'''),
code('''weights = np.full((5, 5), -0.25)
weights[:, 2] = 1.0
w = weights.reshape(-1)
b = -2.0
scores = X @ w + b
print(dict(zip(names, scores)))
assert np.allclose(scores, [3, -2, -3.25, -2])
assert np.isclose(weighted_sum(X[0], w) + b, scores[0])

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
for ax, data, title in zip(axes, [vertical, weights, vertical * weights],
                         ["Input pixels", "Detector weights", "Input × weight"]):
    im = ax.imshow(data, cmap="RdBu", vmin=-1, vmax=1)
    ax.set(title=title, xlabel="Pixel column", ylabel="Pixel row", xticks=[0, 2, 4], yticks=[0, 2, 4])
    for row in range(5):
        for col in range(5):
            ax.text(col, row, f"{data[row, col]:g}", ha="center", va="center",
                    fontsize=9, color="white" if abs(data[row,col]) > .6 else "black")
fig.colorbar(im, ax=axes, shrink=.8, label="Numerical value")
plt.show()'''),
md('''**Pause:** A shifted vertical line receives a negative score. This detector recognizes a location-specific pattern. Seeing two successful examples would not establish that it understands every vertical line.

The bias moves the threshold: with this choice, a raw weighted sum must reach 2 to produce a nonnegative score. Training will later choose weights and biases from examples.

### 4. One matrix multiplication, several detectors
Rows of `X` are examples. Columns of `W` are detectors. We use `X @ W + biases` here.

`(4, 25) @ (25, 2) + (2,)` produces `(4, 2)`.

This is the NumPy convention in this notebook. PyTorch `nn.Linear` stores its weight matrix as `(outputs, inputs)` and uses its transpose for a row-batch input.'''),
code('''W = np.column_stack([weights.reshape(-1), weights.T.reshape(-1)])
biases = np.array([-2., -2.])
batch_scores = X @ W + biases
assert batch_scores.shape == (4, 2)
assert np.allclose(batch_scores[:, 0], scores)
print("             vertical  horizontal")
for name, row in zip(names, batch_scores):
    print(f"{name:16s} {row}")
print("First row equals individual prediction:", np.allclose(X[0] @ W + biases, batch_scores[0]))'''),
md('''The blank image scores equally for both classes. `argmax` would still choose an index. A forced choice is not evidence of confidence.

### 5. Shape reasoning: reductions and broadcasting
For a matrix `(examples, features)`, `mean(axis=0)` reduces the examples axis, leaving one mean per feature. `mean(axis=1)` leaves one mean per example.

Broadcasting compares shapes from the right: corresponding sizes must be equal or one must be 1. Missing leading dimensions act like 1. See [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).'''),
code('''A = np.arange(1, 7, dtype=float).reshape(2, 3)
print("A =", A, sep="\\n")
print("Feature means, axis 0:", A.mean(axis=0))
print("Example means, axis 1:", A.mean(axis=1))
print("Per-feature offsets:", A + np.array([10, 20, 30]), sep="\\n")
print("Per-example offsets:", A + np.array([10, 20])[:, None], sep="\\n")
print("One column:", A[:, 0].shape, "Preserved column axis:", A[:, 0:1].shape)
assert A[:, 0].shape == (2,)
assert A[:, 0:1].shape == (2, 1)'''),
md('''**A bug that runs without an exception:** a target of shape `(n, 1)` subtracted from a prediction of shape `(n,)` produces `(n, n)`. Each target is compared with every prediction.'''),
code('''prediction = np.array([1., 2., 3.])
target_column = np.array([[1.], [2.], [3.]])
wrong_error = prediction - target_column
right_error = prediction - target_column.reshape(-1)
print("Wrong:", wrong_error.shape, "Correct:", right_error.shape)
assert wrong_error.shape == (3, 3)
assert right_error.shape == (3,)
assert np.allclose(right_error, 0)

fig, ax = plt.subplots(figsize=(4.6, 3.5), layout="constrained")
ax.imshow(wrong_error, cmap="RdBu", vmin=-2, vmax=2)
for row in range(3):
    for col in range(3):
        ax.text(col, row, f"{wrong_error[row,col]:g}", ha="center", va="center", color="white" if abs(wrong_error[row,col])==2 else "black")
ax.set(title="Accidental pairwise differences", xlabel="Prediction index", ylabel="Target index", xticks=range(3), yticks=range(3))
plt.show()'''),
md('''### 6. Standardization, without leaking future information
Subtract each feature's training mean and divide by its training standard deviation. Use those same statistics for validation and test examples.

Our synthetic table has two varying features and one constant feature. We define the constant feature's scale as 1 to avoid division by zero. Population standard deviation (`ddof=0`) is used here.'''),
code('''train = np.array([[1., 100., 7.], [2., 200., 7.], [3., 300., 7.], [4., 400., 7.]])
validation = np.array([[5., 500., 7.]])
mean = train.mean(axis=0, keepdims=True)
std = train.std(axis=0, keepdims=True)
scale = np.where(std == 0, 1., std)
train_z = (train - mean) / scale
validation_z = (validation - mean) / scale
print("Training means:", mean)
print("Standardized training data:", train_z, sep="\\n")
print("Validation using TRAINING statistics:", validation_z)
assert np.isfinite(train_z).all()
assert np.allclose(train_z.mean(axis=0), 0)
assert np.allclose(train_z[:, :2].std(axis=0), 1)
assert np.allclose(train_z[:, 2], 0)'''),
md('''The validation row need not have zero mean. Fitting a new normalizer to each test batch changes the model's input meaning.

**Memory matters too:** a basic slice can share memory with its original array. In-place changes can modify both.'''),
code('''original = np.array([1., 2., 3.])
view = original[:2]
view -= 1
print("After editing a view:", original)
independent = original.copy()
independent[:] = 99
assert not np.shares_memory(original, independent)
assert np.array_equal(original, [0, 1, 3])
print("A copy can be changed independently:", independent)'''),
md('''### 7. Matrices move space
A 2×2 matrix tells us where the two basis vectors go. With column vectors we write `T @ point`; with a batch of row vectors we write `points @ T.T`.

Choose `T` below. Try a shear `[[1, 1], [0, 1]]`, a 90° rotation `[[0, -1], [1, 0]]`, or a collapse `[[1, 1], [0, 0]]`.

**Predict:** What happens to the unit square? Can every transformation be undone?'''),
code('''T = np.array([[1., 1.], [0., 1.]])
square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])
transformed = square @ T.T
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, points, title in zip(axes, [square, transformed], ["Original square", f"Transformed · det(T) = {np.linalg.det(T):.1f}"]):
    ax.plot(points[:, 0], points[:, 1], "o-", color=BLUE)
    ax.set(xlim=(-1, 3), ylim=(-1, 3), xlabel="Coordinate 1", ylabel="Coordinate 2", title=title, aspect="equal")
    ax.grid(alpha=.2)
plt.show()
assert np.allclose(transformed[2], T @ square[2])'''),
md('''#### Optional animation: watch the grid shear
Play or drag the frame slider. This interpolation illustrates a transformation, not a learning algorithm. The final frame is the exact matrix `T` above. The static figure remains available if your viewer suppresses JavaScript.'''),
code('''grid_lines = []
for value in np.linspace(-1, 1, 5):
    grid_lines.extend([np.array([[-1., value], [1., value]]), np.array([[value, -1.], [value, 1.]])])
fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal", xlabel="Coordinate 1", ylabel="Coordinate 2")
artists = [ax.plot([], [], color=BLUE, alpha=.5)[0] for _ in grid_lines]
border, = ax.plot([], [], color=ORANGE, lw=3)
title = ax.set_title("")
def animate_transform(frame):
    t = frame / 30
    current = (1-t) * np.eye(2) + t * T
    for artist, points in zip(artists, grid_lines):
        result = points @ current.T
        artist.set_data(result[:, 0], result[:, 1])
    result = square @ current.T
    border.set_data(result[:, 0], result[:, 1])
    title.set_text(f"Matrix interpolation · {t:.0%} of the way to T")
    return artists + [border, title]
animation = FuncAnimation(fig, animate_transform, frames=31, interval=80, blit=False)
animation_html = animation.to_jshtml(default_mode="once")
plt.close(fig)
display(HTML(animation_html))'''),
md('''The shear preserves area here: its determinant is 1. A singular transformation collapses a direction and cannot have an inverse.

Matrix order matters. The rightmost matrix acts first in column-vector notation.'''),
code('''rotation = np.array([[0., -1.], [1., 0.]])
stretch = np.diag([2., 1.])
point = np.array([1., 0.])
print("Rotate then stretch:", stretch @ rotation @ point)
print("Stretch then rotate:", rotation @ stretch @ point)
assert not np.allclose(stretch @ rotation @ point, rotation @ stretch @ point)'''),
md('''## Checks
### Pair exercises · 15 minutes
Replace `None` in each cell. The notebook runs before you solve the exercises, but prints **Not attempted**. A clean Run All by itself does not mean you completed them.

**E1. One batch, three outputs.** Create three detectors that compute the first feature, second feature, and sum of the two. Complete `exercise_W`.'''),
code('''exercise_X = np.array([[1., 2.], [3., 4.]])
exercise_W = None  # TODO: a (2, 3) matrix
if exercise_W is None:
    print("E1: Not attempted")
else:
    assert np.asarray(exercise_W).shape == (2, 3)
    assert np.allclose(exercise_X @ exercise_W, [[1, 2, 3], [3, 4, 7]])
    print("E1: Passed")'''),
md('''**E2. Normalize per example.** Standardize each row of `exercise_X` independently. Keep the row dimension when computing the statistics so broadcasting works. Explain why this differs from the per-feature normalization earlier.'''),
code('''exercise_z = None  # TODO: standardize each row
if exercise_z is None:
    print("E2: Not attempted")
else:
    assert np.asarray(exercise_z).shape == exercise_X.shape
    assert np.allclose(exercise_z, [[-1, 1], [-1, 1]])
    print("E2: Passed")'''),
md('''**E3. Plain Python matrix–vector multiplication.** Implement `matrix_vector` with lists and loops/comprehensions. This prepares the programming technique for Homework 1, without solving its assigned system.'''),
code('''def matrix_vector(matrix, vector):
    # TODO: validate row lengths, then compute one weighted sum per row
    return None

exercise_result = matrix_vector([[1, 2], [3, 4]], [2, -1])
if exercise_result is None:
    print("E3: Not attempted")
else:
    assert exercise_result == [0, 2]
    assert matrix_vector([[0, 1], [1, 0]], [5, 6]) == [6, 5]
    print("E3: Passed")'''),
md('''### Exit ticket
1. What are the shapes of `X`, `W`, and `X @ W` for 10 images of 5×5 pixels and 3 detectors?
2. Why does `*` not replace `@` for a batch of predictions?
3. How could a detector succeed on our centred image and fail on a new image?
4. Which data should determine the normalization statistics?
5. Is a larger raw score automatically a calibrated probability?

## Next Steps
Lab 2 covers NumPy exercises, classification metrics, and Homework 1 (linear systems). **Homework 1 requires Python lists and no NumPy/linear algebra framework.** Follow the published Lab 2 deadline: before Lab 3. Confirm submission/presentation arrangements with your instructor.

For an optional taste of learning, open `LearningPreview.ipynb`.

### Focused resources
- [Current NN course page](https://sites.google.com/view/rbenchea/neural-networks): Course 1 on ML concepts, linear algebra, NumPy.
- [2026 Lab 1](https://github.com/Tensor-Reloaded/Neural-Networks-2026/tree/main/Lab01) and [Lab 2](https://github.com/Tensor-Reloaded/Neural-Networks-2026/tree/main/Lab02).
- [Original Python warm-up](https://github.com/Tensor-Reloaded/Neural-Networks-Template-2025/blob/main/Lab01/Lab01.ipynb), still linked by the 2026 lab.
- 3Blue1Brown: [Linear transformations and matrices](https://www.3blue1brown.com/lessons/linear-transformations/) and [Matrix multiplication as composition](https://www.3blue1brown.com/lessons/matrix-multiplication/). Predict the transformed basis vectors before playing each explanation.
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html): indexing, shape, copies and views.
- [AI Learning Hub prerequisites](https://github.com/Tensor-Reloaded/AI-Learning-Hub/blob/main/foundations/prerequisites.md).

Sources reviewed 2 October 2026. This notebook is supplementary teaching material, not a change to the official assessment rules.''')
], "FirstLab.ipynb")

notebook([
md('''# Optional preview · How do the weights learn?
10–15 minutes for a group already comfortable with arrays. Skip this if the core exercises need time.

## Goal
See the difference between a prediction, a loss, a gradient, and an update. Then see why fitting the known examples perfectly is not enough.

## Setup
Only NumPy and Matplotlib. Synthetic data; no downloads or GPU. This is an extension beyond the published NN Lab 1.'''),
code(SETUP),
md('''## Steps
### 1. An adjustable line
Our model is `prediction = w * x + b`. Both parameters are shared by all examples. The mean squared error measures average squared prediction error.

For residual `r = w*x+b-y`, the gradients are `dw = 2*mean(r*x)` and `db = 2*mean(r)`. We subtract a small multiple of these gradients.'''),
code('''x = np.linspace(-1, 1, 25)
y = 1.5*x + .4 + np.random.default_rng(42).normal(0, .12, len(x))

def loss_and_gradient(w, b):
    residual = w*x + b - y
    return np.mean(residual**2), np.array([2*np.mean(residual*x), 2*np.mean(residual)])

def fit(learning_rate, steps=40):
    parameters = np.array([-1., -1.])
    history = [parameters.copy()]
    losses = [loss_and_gradient(*parameters)[0]]
    for _ in range(steps):
        _, gradient = loss_and_gradient(*parameters)
        parameters -= learning_rate * gradient
        history.append(parameters.copy())
        losses.append(loss_and_gradient(*parameters)[0])
    return np.array(history), np.array(losses)

history, losses = fit(.2)
print(f"MSE: {losses[0]:.4f} → {losses[-1]:.4f}")
print("Learned [w, b]:", history[-1])
assert losses[-1] < losses[0] / 100

# An independent numerical check of the analytic gradient.
parameters = np.array([.7, -.2])
epsilon = 1e-6
finite_difference = []
for direction in np.eye(2):
    high = loss_and_gradient(*(parameters + epsilon*direction))[0]
    low = loss_and_gradient(*(parameters - epsilon*direction))[0]
    finite_difference.append((high-low)/(2*epsilon))
assert np.allclose(finite_difference, loss_and_gradient(*parameters)[1], atol=1e-6)'''),
code('''fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].scatter(x, y, color=BLUE, label="Training examples")
for index, label, style in [(0, "Before training", "--"), (-1, "After training", "-")]:
    w, b = history[index]
    axes[0].plot(x, w*x+b, style, color=ORANGE if index==-1 else "gray", label=label)
axes[0].set(title="A shared rule for every example", xlabel="Input x", ylabel="Output y")
axes[0].legend()
for rate, style, color in [(.02, ":", "gray"), (.2, "-", BLUE), (1.1, "--", ORANGE)]:
    _, trial_losses = fit(rate)
    axes[1].semilogy(trial_losses, style, color=color, label=f"Learning rate {rate}")
axes[1].set(title="Step size changes the learning path", xlabel="Parameter updates", ylabel="Training MSE (log scale)")
axes[1].legend()
plt.show()'''),
md('''For this particular quadratic problem, 0.2 converges and 1.1 diverges. There is no universal best learning rate.

### 2. Watch predictions and loss change together
Drag the frame slider. The left panel is a function of **input x**. The right panel records loss as **parameters change**. The optimizer changes parameters, while the examples stay fixed.'''),
code('''fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].scatter(x, y, color=BLUE)
line, = axes[0].plot(x, history[0,0]*x + history[0,1], color=ORANGE, lw=2)
axes[0].set(xlabel="Input x", ylabel="Output y", ylim=(-2.5, 2.5), title="Current model")
axes[1].plot(losses, color="lightgray")
trail, = axes[1].plot([], [], color=BLUE, lw=2)
axes[1].set(xlabel="Parameter updates", ylabel="Training MSE", xlim=(0, len(losses)-1), ylim=(0, losses.max()*1.08))
def animate_learning(frame):
    w, b = history[frame]
    line.set_ydata(w*x+b)
    trail.set_data(np.arange(frame+1), losses[:frame+1])
    axes[1].set_title(f"Update {frame}: MSE {losses[frame]:.3f}")
    return line, trail
animation = FuncAnimation(fig, animate_learning, frames=len(history), interval=140)
plt.close(fig)
display(HTML(animation.to_jshtml(default_mode="once")))'''),
md('''### 3. Perfect memory can fail on new inputs
Both rules below fit exactly the same three training examples. Their behaviour elsewhere differs. We constructed the second rule to illustrate the problem; no training method is being claimed here.'''),
code('''train_x = np.array([1., 2., 3.])
train_y = 2*train_x
heldout_x = 4.
heldout_y = 8.
def simple_rule(x):
    return 2*x
def interpolating_rule(x):
    return 2*x - (4/3)*(x-1)*(x-2)*(x-3)

assert np.allclose(simple_rule(train_x), train_y)
assert np.allclose(interpolating_rule(train_x), train_y)
grid = np.linspace(.8, 4.1, 200)
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.plot(grid, simple_rule(grid), color=BLUE, label="Simple rule: 2x")
ax.plot(grid, interpolating_rule(grid), "--", color=ORANGE, label="Another rule: fits the same training data")
ax.scatter(train_x, train_y, color="black", label="Training", zorder=4)
ax.scatter([heldout_x], [heldout_y], marker="*", s=160, color=BLUE, label="Held-out example", zorder=4)
ax.set(title="Training fit does not establish generalization", xlabel="Input x", ylabel="Output y")
ax.legend(fontsize=10)
plt.show()
print("At x=4, simple rule:", simple_rule(4), "alternative rule:", interpolating_rule(4))'''),
md('''## Checks
Explain which step computes predictions, which computes gradients, and which changes parameters. Backpropagation is an efficient way to compute gradients in a composed function. An optimizer uses those gradients to update parameters.

Name one way to accidentally use test information while building a model. Validation supports model choices; reserve a final test set for the final evaluation.

## Next Steps
The published NN sequence develops these ideas later. For now, master shapes and weighted sums.

- [3Blue1Brown: neural networks](https://www.3blue1brown.com/lessons/neural-networks)
- [3Blue1Brown: gradient descent](https://www.3blue1brown.com/lessons/gradient-descent)
- [Course 1](https://sites.google.com/view/rbenchea/neural-networks), especially the model/generalization example (slides 17–32).

Optional extension, prepared 2 October 2026. The learning example fits training data only and is not a reported test benchmark.''')
], "LearningPreview.ipynb")

print("Created FirstLab.ipynb and LearningPreview.ipynb")
