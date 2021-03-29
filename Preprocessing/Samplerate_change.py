# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:49:04 2021

@author: CS
"""

# 改变采样率，统一为44.1kHz

import librosa
import numpy as np
import soundfile as sf
import os


file_path = 'D:/Project/DCASE_test/Data/Data_ShipsEar/'
sr_output = 44100
out_path = 'D:/Project/DCASE_test/Data/test/'


file_list = os.listdir(file_path)
for file in file_list:
    wav_path = file_path + file
    data, sr = librosa.load(wav_path, None)
    
    data_output = librosa.resample(data.astype(np.float32), sr, sr_output)
    
    out_name = out_path + file
    sf.write(out_name, data_output, sr_output)
