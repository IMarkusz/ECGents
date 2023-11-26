from scipy.signal import butter, sosfilt, iirnotch, filtfilt
from scipy import ndimage
import numpy as np


def baseline_filter(signal, fs, order=3, **kwargs):
    assert fs > 0, "Sampling frequency cannot be lower than 1"
    assert order > 0, "Filter order cannot be lower than 1"
    if fs < 250:
        raise UserWarning("Sampling frequency should be atleast 250 Hz")
    sos = butter(order, 0.5, "hp", fs=fs, output="sos", analog=False, **kwargs)
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal


def powerline_filter(signal, fs, cut=50):
    assert fs > 0, "Sampling frequency cannot be lower than 1"
    assert cut >= 0, "Cut frequeny cannot be lower than 0"
    if fs < 250:
        raise UserWarning("Sampling frequency should be atleast 250 Hz")
    b, a = iirnotch(cut, 30, fs)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def running_mean_convolve(signal, window, mode="valid"):
    assert window > 0, "window cannot be lower than 1"
    return np.convolve(signal, np.ones(window) / float(window), mode)


def moving_median_detrend(signal, window):
    assert window > 0, "window cannot be lower than 1"
    return signal - ndimage.median_filter(signal, size=window)


def running_mean_cumsum(signal, window):
    assert window > 0, "window cannot be lower than 1"
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def preprocess(x):
    x = baseline_filter(x, fs=500)
    x = powerline_filter(x, fs=500, cut=50)
    x = running_mean_convolve(x, window=10)
    x = moving_median_detrend(x, window=100)

    return x


def strip_lead_trail_zeros(x):
    first_nonzero = np.argmax(x != 0)
    last_nonzero = len(x) - np.argmax(x[::-1] != 0)
    return x[first_nonzero:last_nonzero]
