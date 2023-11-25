#!/usr/bin/env python3
import numpy as np
import sys
import os
from scaleogram import fastcwt
import scaleogram as scg 
from ECGXMLReader.ECGXMLReader import ECGXMLReader
from multiprocessing import Pool
from PIL import Image
import matplotlib.pyplot as plt

DIR_PATH = "/mnt/cluster_vsbud-dl1-005/hackathon"
OUT_DIR_PATH = "/mnt/cluster_vsbud-dl1-004/hackathon/hackathon_wavelet"

LEADS = [
    "I",
    "II",
    "III",
    "aVL",
]

def read_ecgxml(path, lead_order, data_type="short"):
    if data_type == "short":
        i = 0
    elif data_type == "long":
        i = 1
    else:
        raise Exception("wrong data type")
    ecg = ECGXMLReader(path, augmentLeads=True)

    leads = ecg.LeadVoltages[i]

    voltages = []

    for lead in lead_order:
        voltages.append(leads[lead])

    return np.stack(voltages)

def create_wavelet(ecg_data):

    first_nonzero = np.argmax(ecg_data != 0)
    last_nonzero = len(ecg_data) - np.argmax(ecg_data[::-1] != 0)
    ecg_data = ecg_data[first_nonzero : last_nonzero]

    signal_length = len(ecg_data)
    scales = scg.periods2scales( np.arange(1, (signal_length+1)//2) )

    coefs, scales_freq = fastcwt(ecg_data, scales, "morl")
    #coefs_abs = np.abs(coefs)

    return coefs


def ecgxml_to_wavelet(path):

    try:
        xml_filename = path[path.rfind("/")+1:]
        

        voltages = read_ecgxml(path, LEADS, data_type="short")

        wavelets = []
        for voltage in voltages:
            wavelets.append(create_wavelet(voltage))

        # coeffs = np.stack(wavelets)

        for i, wv_transformed_signal in enumerate(wavelets):

            output_filename = xml_filename[:xml_filename.find(".")] + f"_{i}.png"
            output_path = os.path.join(OUT_DIR_PATH, output_filename)

            fig, ax = plt.subplots()
            ax.matshow(np.abs(wv_transformed_signal), cmap="gray")
            ax.axis('off')
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close('all')



        #coeffs_normalized = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min())

        #color_values = (coeffs_normalized * 255).astype(np.uint16)

        #img = Image.fromarray(color_values.transpose(1, 2, 0), 'RGBA')

        # output_path = os.path.join(OUT_DIR_PATH, output_filename)

        # print(output_path)
        # img.save(output_path)

        # output_filename = xml_filename[:xml_filename.find(".")] + "_matplotlib.png"

        # plt.matshow(coeffs[0])
        # plt.savefig(output_filename)

    except:
        pass



if __name__ == '__main__':

    ecgxml_paths = [os.path.join(DIR_PATH, path) for path in os.listdir(DIR_PATH)]

    with Pool(90) as p:
        p.map(ecgxml_to_wavelet, ecgxml_paths)