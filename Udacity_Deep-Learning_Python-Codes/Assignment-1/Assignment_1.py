from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
import pickle

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    #A hook to report the progress of a download. This is mostly intended for users with
    #slow internet connections. Reports every 1% change in download progress.
    global last_percent_reported
