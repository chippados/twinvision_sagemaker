import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
import argparse

def model_fn(model_dir):
    """Load the model for inference"""
    model = load_model(os.path.join(model_dir, '1'))
    return model