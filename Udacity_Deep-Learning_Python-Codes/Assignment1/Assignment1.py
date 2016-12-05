from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from urllib.request import urlretrieve
from pathlib import Path
import pickle

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
dirname = os.path.dirname
dataPath = dirname(dirname(os.getcwd()))+'\\DataSets\\'

# Download progress hook function
def download_progress_hook(count, blockSize, totalSize):
    #A hook to report the progress of a download. This is mostly intended for users with
    #slow internet connections. Reports every 1% change in download progress.
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent %5 ==0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent

#Download the data sets
def maybe_download(filename, expected_bytes, force=False):
    #Download a file if not present, and make sure it's the right size.
    filePath = dataPath + filename

    if force or not os.path.exists(filePath):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filePath, reporthook = download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filePath)

    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043) 

#Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labelled A through J.
num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force = False):
    filePath = dataPath + filename
    root = os.path.splitext(os.path.splitext(filePath)[0])[0] # remove .tar.gz
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s. ' %(root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filePath)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

#Problem 1 Display images
#image_files = os.listdir(train_folders[0])

#img = mpimg.imread(os.path.join(train_folders[0], image_files[0]))
#plt.imshow(img,cmap='gray')
#plt.show()

image_size = 28     # image width&height
pixel_depth = 255.0 # max intensity

def load_letter(folder, min_num_images):
    # load the data for a single letter
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),dype=np.float32)
    print(folder)    
    num_images = 0

    for img in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) - pixel_depth/2)/pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read: ', image_file, ':', e, ' - it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]   #rescale the dataset to the correct size
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor: ', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force = False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to ', set_filename, ':', e)
    return dataset_names


