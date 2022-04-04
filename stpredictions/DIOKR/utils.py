import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_compounds_info(n, data_folder=None):
    """
    Load the compounds_info.txt file with the columns: (INCHIKEY, MOLECULAR_FORMULA, CV_FOLD)

    Parameters
    ------------
    - n : int
        Number of fingerprints to load
    - data_folder : str
        Path of folder containing the data
    """

    # if data_folder is None:
    #     data_folder = Path("Implementation/data/Donnees_metabolites/")
    file_to_open = data_folder + "compounds_info.txt"
    data = pd.read_csv(file_to_open, sep="\t", engine='python', nrows=n)
    return data

def load_candidates(formula, n_max, random=False):

    # load candidates for a given molecular formula
    path_to_candidates = 'C:/Users/user/Documents/deep_kernel_learning_code/Data/metabolites/Candidats/'

    try:
        with open(path_to_candidates + formula + '.csv') as f:
            candidates = f.readlines()
    except FileNotFoundError:
        return -1, -1
    # return error if too many candidates or zero candidate
    if len(candidates) == 0 or len(candidates) > n_max:
        return -1, -1

    # Parse load data to compute lists of inchikeys and fingerprints
    fingerprints = []  # will contain the fingerprints of the candidates
    inchikey_fingerprints = []  # will contain the inchikey of the candidates
    if random is False:
        for c in range(len(candidates)):
            inchikey_fingerprints.append(candidates[c].split('\t')[0])
            f = np.zeros(7593).astype(int)
            f[np.array(candidates[c].split('\t')[1:]).astype(int)] = 1
            fingerprints.append(f)

    else:
        random_idx = np.random.randint(len(candidates))
        inchikey_fingerprints.append(candidates[random_idx].split('\t')[0])
        f = np.zeros(7593).astype(int)
        f[np.array(candidates[random_idx].split('\t')[1:]).astype(int)] = 1
        fingerprints.append(f)

    cf = np.array(fingerprints)

    return cf, inchikey_fingerprints

def load_data_metabolites(path):
    """
    Load metabolites data (X: mass spectrums, Y: fingerprints)

    Parameters
    ----------
    path : str
        Path of the folder containing mass spectrums and fingerprints.

    Returns
    -------
    None.

    """
    path_X = path + 'spectra_as_vec_pos.mat'
    spectrums = loadmat(path_X)
    X = spectrums['spectra_as_vec'].toarray()
    
    idx = np.argwhere(np.all(X[..., :] == 0, axis=0))
    data = np.delete(X, idx, axis=1)
    X = data
    
    path_Y = path + 'all_fingerprints.mat'
    fingerprints = loadmat(path_Y)
    Y = fingerprints['Y'].T
    
    return X, Y
    