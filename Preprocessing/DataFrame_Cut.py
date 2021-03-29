# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:41:06 2020

@author: CS
"""
# 2021年3月18日 
# 数据集分帧，10s截取
# 按照DCASE 2020 task1的方式进行命名

from pydub import AudioSegment
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import librosa.display



def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size)
        start = int(start)

# 读入所有wav文件
def read_wav(file_path):
    wave_list = []
    file_list = os.listdir(file_path)
    for file_list in file_list:
        # splitext 分离文件名与扩展名
        name, category = os.path.splitext(file_path + file_list)
        if category == '.wav':
            wave_list.append(file_list)
    return wave_list
        

# # 显示音频时域波形
# for wav in wave_list[1:3]:    
#     y, sr = librosa.load(wav)
#     plt.figure(figsize=(12, 8))
#     plt.subplot(4, 2, 1)
#     librosa.display.waveplot(y, sr=sr)
    


def data_cut(file_save, wav_list):
    for wav in wave_list:
        # sound,sr = librosa.load(file_path + wav)
        sound = AudioSegment.from_wav(file_path + wav)
        i = 0
        for (start,end) in windows(sound,window_size):
            if(len(sound[start:end]) == window_size):
                i = i + 1
                sound_clip = sound[start:end]
                
                # 命名
                wav_num = "{0:03d}".format(i)
                
                # ShipsEar
                # sound_clip.export(file_save + wav.split('__')[1].split('.')[0] + 'A' + '-' + 'ShipsEar' + 
                #                   '-' + wav.split('__')[0] + '-' + wav_num + '-' + 'a' + '.wav', format='wav')

                # Yantai
                sound_clip.export(file_save + wav.split('_')[1] + '-' + 'Yantai' + 
                                  '-' + wav.split('_')[0] + '-' + wav_num + '-' + 'a' + '.wav', format='wav')

                # Hai
                # sound_clip.export(file_save + 'Merchant' + 'C' + '-' + 'Hai' + 
                #                   '-' + wav.split('.')[0] + '-' + wav_num + '-' + 'a' + '.wav', format='wav')
                
                
                                      
        print ('The file of ' + str(wav) +  ' has done')
                                

if __name__ == '__main__':
    # 路径
    file_path = 'D:/Project/DCASE_test/Data/Data_Yantai/'
    file_save = 'D:/Project/DCASE_test/Data/test/'
    # 截取长度 ms
    window_size = 10000
    # 读入文件
    wave_list = read_wav(file_path)
    # 按照固定时长进行截取
    data_cut(file_save, wave_list)
    
    
       

     
     
     
   
     
     
     