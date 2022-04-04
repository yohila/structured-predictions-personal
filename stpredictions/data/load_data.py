import numpy as np
import csv
import h5py
from scipy.io import arff
import pandas as pd
import scipy.io as sio
from skmultilearn.dataset import load_from_arff
# from PIL import Image
from numpy import asarray
import os

def load_bibtex():
# def load_bibtex(path='Bibtex/Bibtex_data.txt'):
    """
        Load Dataset Bibtex (d = 1836, p = 159, n_tr = 4880, n_te = 2515, n_tot= 7395)
    """
    this_dir, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir, "bibtex", "bibtex.arff")

    # with open(path, 'r') as file:
    with open(DATA_PATH, 'r') as file:
        data = file.read()
    data = data.split('\n')
    n, p, d = list(map(int, data[0].split(' ')))
    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        z = data[i + 1].split(' ')
        y_idx = list(map(int, z[0].split(',')))
        Y[i, y_idx] = 1
        x_idx = z[1:]
        x_idx = list(map(lambda s: int(s.split(':')[0]), x_idx))
        X[i, x_idx] = 1

    return X, Y, n, p, d

def load_bibtex_train():
    """
        Load Dataset Bibtex (d = 1836, p = 159, n_tr = 4880, n_te = 2515, n_tot= 7395)
    """
    this_dir, this_filename = os.path.split(__file__)
    path_tr = os.path.join(this_dir, "bibtex", "bibtex-train.arff")

    x_train, y_train = load_from_arff(path_tr, label_count=159)

    return x_train, y_train

def load_bibtex_test():
    """
        Load Dataset Bibtex (d = 1836, p = 159, n_tr = 4880, n_te = 2515, n_tot= 7395)
    """
    this_dir, this_filename = os.path.split(__file__)
    path_te = os.path.join(this_dir, "bibtex", "bibtex-test.arff")

    x_test, y_test = load_from_arff(path_te, label_count=159)

    return x_test, y_test


def load_delicious(path='Delicious/Delicious_data.txt'):
    """
        Load Dataset Delicious
    """

    with open(path, 'r') as file:
        data = file.read()
    data = data.split('\n')
    n, p, d = list(map(int, data[0].split(' ')))
    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        z = data[i + 1].split(' ')
        if z[0] != '':
            y_idx = list(map(int, z[0].split(',')))
            Y[i, y_idx] = 1
        x_idx = z[1:]
        x_idx = list(map(lambda s: int(s.split(':')[0]), x_idx))
        X[i, x_idx] = 1

    return X, Y, n, p, d


def load_yeast(path='YEAST_Elisseeff_Weston_2002.csv'):
    """
        Load Dataset YEAST (p = 103, d = 14, n_tr = 1,500, n_te=917)
    """

    p = 103
    d = 14
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = list(reader)

    n = len(data)

    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        X[i] = np.array(data[i][:p])
        Y[i] = np.array([True if data[i][p + j] == 'TRUE' else False for j in range(d)])

    return X, Y, n, p, d


def load_scene(path='Data/Scene/scene_train'):
    if path[-1] == 'n':
        n, p, d = 1211, 294, 6
    else:
        n, p, d = 1196, 294, 6

    with open(path, 'r') as file:
        data = file.read()

    data = data.split('\n')

    X = np.zeros((n, p))
    Y = np.zeros((n, d))

    for i in range(n):
        z = data[i].split(' ')
        y_idx = list(map(int, z[0].split(',')))
        Y[i, y_idx] = 1
        x = z[1:]
        x = list(map(lambda s: float(s.split(':')[1]), x))
        X[i, :] = x

    return X, Y, n, p, d


def load_usps(path='Data/usps.h5'):
    """
        (7291, 256) (7291,) (2007, 256) (2007,)
    """

    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    return X_tr, y_tr, X_te, y_te


def load_cal500(path='../Data/CAL500/CAL500.arff'):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, 0:68].to_numpy(), df.iloc[:, 68:].to_numpy().astype(int)

    return X, Y


def load_emotions(path='../Data/emotions/emotions.arff'):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, 0:72].to_numpy(), df.iloc[:, 72:].to_numpy().astype(int)

    return X, Y


def load_birds(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, 0:260].to_numpy(), df.iloc[:, 260:].to_numpy().astype(int)

    return X, Y


def load_enron(path='../Data/enron/enron.arff'):
    X, Y = load_from_arff(path, label_count=53)
    X, Y = X.todense(), Y.todense()

    return X, Y

def load_mediamill(path):

    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, 0:120].to_numpy(), df.iloc[:, 120:].to_numpy().astype(int)

    return X, Y

def load_corel5k(path):

    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, 0:499].to_numpy(), df.iloc[:, 499:].to_numpy().astype(int)

    return X, Y

def load_jrcacquis(path='../Data/jrcacquis/dataset/KX_de.mat'):
    K_x = sio.loadmat(path)['KX']

    return K_x

def load_horses(path, ravel=True, dim=32):
    n = 328
    RGB = []
    FG = []
    for i in range(n):
        rgb_path = path + '/rgb/horse' + str(i + 1).zfill(3) + '.jpg'
        rgb = Image.open(rgb_path)
        rgb_resized = rgb.resize((dim, dim))
        rgb_array = asarray(rgb_resized)
        fg_path = path + '/figure_ground/horse' + str(i + 1).zfill(3) + '.jpg'
        fg = Image.open(fg_path)
        fg_resized = fg.resize((dim, dim))
        fg_array = asarray(fg_resized)
        if not ravel:
            RGB.append(rgb_array)
            FG.append(fg_array)
        else:
            RGB.append(rgb_array.ravel())
            FG.append(fg_array.ravel())

    RGB = np.array(RGB)
    FG = np.array(FG)

    return RGB, FG


def load_mulan(path):
    """
        Load Mulan Dataset
    """

    with open(path, 'r') as file:
        data = file.read()
    data = data.split('\n')
    n, p, d = list(map(int, data[0].split(' ')))
    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        z = data[i + 1].split(' ')
        if len(z[0]) > 0:
            y_idx = list(map(int, z[0].split(',')))
        else:
            y_idx = []
        Y[i, y_idx] = 1
        x_idx = z[1:]
        x_values = list(map(lambda s: float(s.split(':')[1]), x_idx))
        x_idx = list(map(lambda s: int(s.split(':')[0]), x_idx))
        X[i, x_idx] = x_values

    return X, Y, n, p, d

def load_mulan_arff(path):

    X, Y = load_from_arff(path, label_count=208)
    X, Y = X.todense(), Y.todense()

    return X, Y