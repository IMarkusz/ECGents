import pickle

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from spectral_ai import AlexNetModel
from preprocess import strip_lead_trail_zeros
from ECGXML_to_wavelet import ecgxml_to_wavelet


code2label = ["cardiology clinic", "other", "outpatient"]
label2code = {"cardiology clinic": 0, "other": 1, "outpatient": 2}


def analyze(path):
    model = AlexNetModel()
    with open("models/pca.pickle", "rb") as f:
        pca = pickle.load(f)
    with open("models/knn.pickle", "rb") as f:
        knn = pickle.load(f)

    wavelets = ecgxml_to_wavelet
