# Lab 5 · Image Classification with a Convolutional Neural Network (MNIST)

From single convolutional layers to a LeNet-style CNN trained and evaluated on MNIST.

**File:** [`05_cnn_lenet_mnist_classification.py`](05_cnn_lenet_mnist_classification.py). It is a Colab notebook exported as a Python script, so the Markdown cells appear as comments and docstrings.

## What I worked on

- How `nn.Conv2d` changes the spatial dimensions, and using padding to keep them
- Max pooling and its effect on memory: comparing feature-map sizes and forward time with and without pooling on 512×512 inputs
- Writing a reusable `ConvBlock` (Conv2d → ReLU → MaxPool2d)
- Building a pseudo-LeNet: `ConstantPad2d` (28→32 px), conv blocks and an MLP head `[400 → 120 → 84 → 10]` with `LogSoftmax`
- Training with RMSprop and the NLL loss, and evaluating test loss and accuracy per epoch on GPU

## Stack

Python · PyTorch · torchvision · Matplotlib · NumPy · Google Colab

## How to run

Open the script in [Google Colab](https://colab.research.google.com/) or run it locally:

```bash
pip install torch torchvision numpy matplotlib
python 05_cnn_lenet_mnist_classification.py
```

## Credits

Lab template created by Santiago Pascual and Xavier Giró-i-Nieto (UPC School, 2019), based on a Keras version by Míriam Bellver and updated by Daniel Fojo as course material. The original authors are credited at the top of the file. The exercise solutions are my own work.

---
Part of my [GDSA labs](https://github.com/Pedrom-GH/GDSA_PEDRO) · B.Sc. in Telecommunications Systems Engineering, UPC – ESEIAAT · Pedro Muñoz Álvarez
