import tensorflow as tf      # Deep Learning library
import numpy as np 

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

def preprocess_frame(frame):
    normalized_frame = frame/255.0    
    return preprocessed_frame

