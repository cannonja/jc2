import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sig(x):
    sig = sigmoid(x)

    return sig * (1 - sig)
