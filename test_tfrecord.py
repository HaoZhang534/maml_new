import csv
import numpy as np
import pickle
import random
import tensorflow as tf
from tqdm import trange
from data_generator_rcd import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from math import isnan

dataset = tf.data.TFRecordDataset(['train_own.tfrecords'])

def dgcae_example_feature(example_proto):
    features={}