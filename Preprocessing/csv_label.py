import os
import csv
import random

filePath = './Data/TAU-urban-acoustic-scenes-2020-mobile-development/audio/'
def readname():
    name = os.listdir(filePath)
    return name
name = readname()
count_name= len(name)
count_60=int(0.6*count_name)
count_80=int(0.8*count_name)
random.seed(123)
random.shuffle(name)

f1 = open('meta.csv', 'w', encoding='utf-8', newline='')  # newline 删除空行
csv_writer1 = csv.writer(f1, delimiter='\t')  # 设置分隔符，默认为用逗号分隔，这里用'\t'
csv_writer1.writerow(["filename", "scene_label", "identifier", "source_label"])  # 写表头

f2 = open('fold1_train.csv', 'w', encoding='utf-8', newline='')  # newline 删除空行
csv_writer2 = csv.writer(f2, delimiter='\t')  # 设置分隔符，默认为用逗号分隔，这里用'\t'
csv_writer2.writerow(["filename", "scene_label"])  # 写表头

f3 = open('fold1_test.csv', 'w', encoding='utf-8', newline='')  # newline 删除空行
csv_writer3 = csv.writer(f3, delimiter='\t')  # 设置分隔符，默认为用逗号分隔，这里用'\t'
csv_writer3.writerow(["filename"])  # 写表头

f4 = open('fold1_evaluate.csv', 'w', encoding='utf-8', newline='')  # newline 删除空行
csv_writer4 = csv.writer(f4, delimiter='\t')  # 设置分隔符，默认为用逗号分隔，这里用'\t'
csv_writer4.writerow(["filename", "scene_label"])  # 写表头

f5 = open('fold1_inference.csv', 'w', encoding='utf-8', newline='')  # newline 删除空行
csv_writer5 = csv.writer(f5, delimiter='\t')  # 设置分隔符，默认为用逗号分隔，这里用'\t'
csv_writer5.writerow(["filename", "scene_label"])  # 写表头

count=1
for i in name:
    #     print(i)
    name_str=i.split(sep="-")
    audio_name= 'audio/' + i
    scene_label=name_str[0]
    identifier=name_str[1]+'-'+name_str[2]
    source_label=name_str[4].split(sep=".")[0]
    csv_writer1.writerow([audio_name, scene_label, identifier, source_label])
    if count<=count_60:
        csv_writer2.writerow([audio_name, scene_label])
    elif count_60< count <= count_80:
        csv_writer3.writerow([audio_name])
        csv_writer4.writerow([audio_name, scene_label])
    elif count > count_80:
        csv_writer5.writerow([audio_name, scene_label])
    count=count+1

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
