import numpy as np
import Layers
import ActivationFunction as AcFun
import Loss
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Create dataset
X, y = spiral_data(samples=100, classes=3)
