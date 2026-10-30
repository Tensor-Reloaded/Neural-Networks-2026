## Lab 5 · Activations, optimizers and dropout

### Start here · practical notebook

[**Open the notebook on GitHub**](Lab05.ipynb) · [**Run in Google Colab**](https://colab.research.google.com/github/Tensor-Reloaded/Neural-Networks-2026/blob/main/Lab05/Lab05.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tensor-Reloaded/Neural-Networks-2026/blob/main/Lab05/Lab05.ipynb)

Save a copy in Drive, then choose **Runtime → Run all**. The notebook runs on **CPU**, needs no dataset download, and includes saved plots for reading directly on GitHub. In Colab, PyTorch, NumPy and Matplotlib are already available. For local use: `pip install torch numpy matplotlib ipykernel`.

| Order | Topic | Visual experiment |
|---|---|---|
| 1 | Sigmoid vs ReLU and Leaky ReLU | Activation curves, derivatives and gradients through a chain of layers |
| 2 | SGD, momentum and Adam | Manual update rules, steps on a parabola and paths through the Rosenbrock valley |
| 3 | Inverted dropout | Random training masks, rescaling and deterministic evaluation |
| 4 | Train a small MLP | Same initial weights and data, three dropout rates, train/validation curves and fitted functions |
| 5 | Your experiment | Change one setting, compare results and connect the findings to Homework 3 |

Designed for **two hours**, including discussion and student experiments. Adapted from [Cristian's Lab06/2025](https://github.com/Tensor-Reloaded/Neural-Networks-Template-2025/blob/8be443f/Lab06/Lab06.ipynb). The notebook uses autograd for demonstrations; Homework 3's own implementation requirements still apply. Dropout and optimizer comparisons are illustrative single-seed experiments, not guarantees that one setting always wins.

Related course: [Course 5 — activation functions, initialization, overfitting and regularization](https://drive.google.com/file/d/1Miafi_98lceBEcO5rIGS1xKxyrBT2AGZ/view).


***
Lab Resources:
* [SGD implementation](sgd.py)
* [Backprop, optimizers, learning rate schedulers, batch size schedulers](Backprop,%20optimizers,%20learning%20rate%20schedulers,%20batch%20size%20schedulers.pdf)


***
Homework 3:
- Kaggle competition link: https://www.kaggle.com/t/335cc92e0ed644d39c4a18a6b22bba78
- Deadline: Submit the assignment before Lab 7
***
For self-study (all students):
* [Neural Networks (chapter 1 - chapter 4)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (animated introduction to neural networks and backpropagation)
* [The Misconception that Almost Stopped AI [How Models Learn Part 1]](https://www.youtube.com/watch?v=NrO20Jb-hy0)
* [The F=ma of Artificial Intelligence [Backpropagation, How Models Learn Part 2]](https://www.youtube.com/watch?v=VkHfRKewkWw)
* [Why Deep Learning Works Unreasonably Well [How Models Learn Part 3]](https://www.youtube.com/watch?v=qx7hirqgfuU)
***
