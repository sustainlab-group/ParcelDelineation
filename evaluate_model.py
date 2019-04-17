# Run python evaluate_model.py [path to array of predictions] [path to test/validation dataframe]
import sys
from PIL import Image
from random import randint
import numpy as np
import pandas as pd
import math
import warnings
import pdb
from matplotlib import pyplot as plt
from utils.metrics import get_metrics
from utils.data_loader_utils import read_imgs_keraspp, read_imgs_keraspp_stacked

if len(sys.argv) != 3:
	print("You should run: python evaluate_model.py [path to array of predictions] [path to test dataframe]")
	sys.exit()

predictions = np.load(sys.argv[1])
test_df = pd.read_csv(sys.argv[2]) 
x_true, y_true = read_imgs_keraspp(test_df)
y_true = y_true.flatten()
y_pred = predictions.flatten()

get_metrics(y_true, y_pred, binarized=False)

