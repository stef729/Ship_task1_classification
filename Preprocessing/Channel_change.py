# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:42:56 2021

@author: CS
"""

# 将双通道的wav转换为单通道

from scipy.io import wavfile
import pandas as pd
import numpy as np
import os
import librosa
import soundfile as sf


file_path = 'D:/Project/Sound/Chuan/data/'
out_path = 'D:/Project/DCASE_test/Data/test/'
samplerate = 44100


file_list = os.listdir(file_path)
for file in file_list:
    wav_path = file_path + file
    y, sr = librosa.load(wav_path, mono=False)
    
    y_mono = librosa.to_mono(y)
    
    out_name = out_path + file
    sf.write(out_name, y_mono, samplerate)
    
    
    
    
    
    
    
    
    
    # sample_rate, wav_data = wavfile.read(wav_path)
    
    # if wav_data.ndim > 1:
    #     sound_array = np.array(wav_data)
    #     sound = np.mean(sound_array, axis=1)
    #     sound2 = sound.astype(int)
    #     wavfile.write('D:/Project/DCASE_test/Data_Hai/' + file_list, 44100, sound2)
    # else:
    #     wavfile.write('D:/Project/DCASE_test/Data_Hai/' + file_list, 44100, wav_data)

