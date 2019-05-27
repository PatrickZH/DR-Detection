import numpy as np
import os
import csv
import h5py
import PIL.Image as Image
import torchvision.transforms as transforms

# from multiprocessing.dummy import Pool as ThreadPool
# import time
#
# num = 500000
# images_all = np.zeros(shape=(num, 10, 10, 3))
#
# t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
# print('%s begin' % t_c)
#
# for i in range(num):
#     # np.random.seed(i)
#     images_all[i] = np.random.rand(10, 10, 3)
#
# t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
# print('%s end' % t_c)
#
# # print('mean: ', np.mean(images_all[:,:,:,0]), np.mean(images_all[:,:,:,1]), np.mean(images_all[:,:,:,2]))
#
#
#
#
#
#
# t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
# print('%s begin' % t_c)
#
# def fun(idx):
#     # np.random.seed(idx)
#     images_all[idx] = np.random.rand(10, 10, 3)
#
#
# # Make the Pool of workers
# pool = ThreadPool()
# idxs = list(range(num))
# pool.map(fun, idxs)
# #close the pool and wait for the work to finish
# pool.close()
# pool.join()
#
# t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
# print('%s end' % t_c)
# # print('mean: ', np.mean(images_all[:,:,:,0]), np.mean(images_all[:,:,:,1]), np.mean(images_all[:,:,:,2]))

# x = str('abc').encode('utf-8')
# print(x)
#

images = []
labels = []
notes = []


with open('D:/eye/kaggle/labelme_new_sorted.csv', "r", encoding='utf-8') as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for i, each in enumerate(dict_reader):
        images.append(each['image'])
        labels.append('')
        notes.append('')





with open('D:/eye/kaggle/眼底图像标注LM.csv', "r", encoding='utf-8') as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for i, each in enumerate(dict_reader):
        keys = list(each.keys())
        image = each[keys[0]]
        if '-' in image:
            tokens = image.split('-')
            id = tokens[0]
            side = tokens[1]
        elif '_' in image:
            tokens = image.split('_')
            id = tokens[0]
            side = tokens[1]
        else:
            id = image[:-1]
            side = image[-1]
        if side == 'L':
            image = tokens[0]+'_left.jpeg'
        elif side == 'R':
            image = tokens[0]+'_right.jpeg'
        else:
            print('error', image)
            continue

        label = each[keys[1]]
        note = each[keys[2]]
        # print(image, label, note)

        if image in images:
            idx = images.index(image)
            labels[idx] = label
            notes[idx] = note


csvFile = open("../kaggle/labelme_LM_20190418.csv", "w", newline='')
writer = csv.writer(csvFile)
fileHeader = ["image", "level", "note"]
writer.writerow(fileHeader)
ids = []
for id in range(len(images)):
    writer.writerow([images[id], labels[id], notes[id] ])

csvFile.close()




        # image = each['image'] + '.jpeg'
        # level = int(each['level'])
        # id = int(each['image'].split('_')[0])
        # ids.append(id)
        # print(image)

    # ids = list(set(ids))
    # ids.sort()
    # print(ids)


# csvFile = open("../kaggle/labelme_new_sorted.csv", "w", newline='')
# writer = csv.writer(csvFile)
# fileHeader = ["image", "level"]
# writer.writerow(fileHeader)
# ids = []
# for id in ids:
#     li = str(id)+'_left.jpeg'
#     writer.writerow([li])
#     ri = str(id)+'_right.jpeg'
#     writer.writerow([ri])
#
# csvFile.close()
#

