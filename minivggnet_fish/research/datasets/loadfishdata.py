from .loader import load_batch
from keras import backend as K
import numpy as np
import os

def load_data():

    path = "/mnt/disseration_work/final_dataset"
    num_train_samples = 9025
    x_train = np.zeros((num_train_samples, 3, 229, 229), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')
