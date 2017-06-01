# ForwardThinking
Companion code for _Forward Thinking: Building and Training Neural Networks One Layer at a Time_ and _Forward Thinking: Building Deep Random Forests_ submitted to NIPS 2017.

Authors:
- Chris Hettinger
- Tanner Christensen
- Ben Ehlert
- Jeffrey Humpherys
- Tyler Jarvis
- David Kartchner
- Kevin Miller
- Sean Wade

## Publications
_Forward Thinking: Building and Training Neural Networks One Layer at a Time_: (URL coming soon)
_Forward Thinking: Building Deep Random Forests_: https://arxiv.org/abs/1705.07366

## Dependencies
- numpy==1.11.3
- tensorflow-gpu==1.0.0
- keras==2.0.4
- matplotlib==2.0.0

## Hardware
We used a single desktop computer with:
- Intel i5-7400 processor
- Nvidia GeForce GTX 1060 3GB GPU
- 8GB DDR4 RAM
With our configuration, it took approximately 2 hours to run the `run_mnist_cnn.sh` script.

## Installation
TODO: Make this pip installable.

## Execution
Once the package has been installed, you may run the included `run_mnist_cnn.sh` script. 

This will run the forward thinking neural network (achieved 99.72% in our tests), the backpropagation equivalent of our model (achieved 99.63% in our tests), and saves and displays a plot comparing the test and train accuracies.
