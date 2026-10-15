See [Backprop](./Backprop.pdf)

### Saturation Issue and Zero Gradient

After computing the forward pass, we observed that the output of the sigmoid function at the output neuron was $y_3 = 1$. The derivative of the sigmoid function is 0  (if we were to use the actual sigmoid value there, this would not be the case). In the general case, while sigmoids work very well as output activations paired with BCE, as hidden layer activations, they can slow down training when we have very small or very large pre-activation values. 

This results in a zero gradient at the output neuron, causing the weight updates to become zero, effectively halting the training process. This issue arises because the sigmoid function saturates near 0 and 1, leading to vanishing gradients.

To mitigate this, try to clip values for the activation input, alternative activation functions such as ReLU or different initialization strategies for weights can help avoid saturation and allow for more effective training.
