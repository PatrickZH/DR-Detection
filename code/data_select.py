import numpy as np
import os
import shutil
import csv

root_path = 'D:/eye/kaggle'
datapath = 'D:/eye/kaggle/train'
trainval_id_path = os.path.join(root_path, 'trainLabels.csv').replace('\\', '/')
targetpath = 'D:/eye/kaggle/selected_new'
id_list = list()
image_list = list()
level_list = list()
ratio = [750, 2000, 1750, 250, 250]
with open(trainval_id_path, "r") as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for i, each in enumerate(dict_reader):
        # if i == 1000:
        #     break
        image = each['image'] + '.jpeg'
        id = str(each['image']).split('_')[0]
        level = int(each['level'])
        id_list.append(id)
        image_list.append(image)
        level_list.append(level)

idx_rand = np.random.permutation(len(image_list))
list_level = [[], [], [], [], []]
list_level_all = []
for i in range(len(id_list)):
    level = level_list[i]
    if id_list[i] not in list_level_all:
        list_level[level].append(id_list[i])
        list_level_all.append(id_list[i])
list_id_selected = []
for i in range(5):
    np.random.shuffle(list_level[i])
    list_id_selected = list_id_selected + list_level[i][:int(ratio[i]/2)]

images = []
for each in list_id_selected:
    images.append(each+'_left.jpeg')
    images.append(each+'_right.jpeg')
print('images: ', len(images))
for i, each in enumerate(images):
    if i %100 == 0:
        print(i)
    shutil.copy(datapath+'/'+each, targetpath+'/'+each)

# 0 - 15%
# 1 - 40%
# 2 - 35%
# 3 - 5%
# 4 - 5%


#
#
#
# files = os.listdir(datapath)
# selected_files = []
# for each in files:
#     tokens = each.split('_')
#     if tokens[0] not in selected_files:
#         selected_files.append(tokens[0])
#
# np.random.shuffle(selected_files)
# selected_files = selected_files[:2500]
# images = []
# for each in selected_files:
#     images.append(each+'_left.jpeg')
#     images.append(each+'_right.jpeg')
#
# for i, each in enumerate(images):
#     if i %100 == 0:
#         print(i)
#     shutil.copy(datapath+'/'+each, targetpath+'/'+each)
#
targetpath = 'D:/eye/kaggle/selected_new'
files = os.listdir(targetpath)
csvFile = open("labelme_new.csv", "w", newline='')
writer = csv.writer(csvFile)
fileHeader = ["image", "level"]
writer.writerow(fileHeader)

for each in files:
    tokens = each.split('.')
    name = tokens[0]
    d1 = [name]
    writer.writerow(d1)

csvFile.close()