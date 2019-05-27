import h5py

import csv
import os
import numpy as np
from PIL import Image
import time

root_path = '../kaggle'
imgpath_trainval = os.path.join(root_path, 'train_crop_512').replace('\\', '/')
imgpath_test = os.path.join(root_path, 'test_crop_512').replace('\\', '/')
annotationpath = os.path.join(root_path, 'trainLabels.csv').replace('\\', '/')
test_id_path = os.path.join(root_path, 'retinopathy_solution.csv').replace('\\', '/')

# threshold = 0.8
num_error = 0
files_trainval = os.listdir(imgpath_trainval)
images_train = []
labels_train = []
images_trainval = []
labels_trainval = []
images_val = []
labels_val = []
with open(annotationpath, "r") as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for each in dict_reader:
        image = each['image'] + '.jpeg'
        level = int(each['level'])
        if image in files_trainval:
            images_trainval.append(image)
            labels_trainval.append(level)
        else:
            num_error += 1
print('errors: %d in trainval data' % num_error)


num_error = 0
files_test = os.listdir(imgpath_test)
images_test = []
labels_test = []
with open(test_id_path, "r") as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for each in dict_reader:
        image = each['image'] + '.jpeg'
        level = int(each['level'])
        if image in files_test:
            images_test.append(image)
            labels_test.append(level)
        else:
            num_error +=1
print('errors: %d in test data'%num_error)



''' pack into one file.h5'''
# package_path = 'D:/eye/kaggle'
# width, height = 512, 512
# images = []
# images_list = []
# labels_list = []
# modes = ['trainval']#['trainval', 'test']
# num_trainval = len(labels_trainval)
# num_test = len(labels_test)
#
# for mode in modes:
#     print(mode)
#     if mode == 'trainval':
#         num = num_trainval
#         impath = imgpath_trainval
#     else:
#         num = num_test
#         impath = imgpath_test
#     f = h5py.File(os.path.join(package_path, 'data_%d_%s_crop.h5' % (width, mode)), 'w')
#
#     impath = str(impath).encode('utf8')
#
#     for i in range(num):
#         if i%100 == 0:
#             print('%d/%d'%(i, num))
#         if mode == 'trainval':
#             image = images_trainval[i]
#             label = labels_trainval[i]
#         else:
#             image = images_test[i]
#             label = labels_test[i]
#         image = str(image).encode('utf8')
#         images_list.append(image)
#         labels_list.append(label)
#         with Image.open(os.path.join(impath, image)) as img:
#             img = img.convert('RGB')
#             # img = img.resize((width, height)) ###################################################################################
#             img = np.array(img, dtype=np.uint8)
#             # images.append(img)
#             f[image] = img
#
#     f['images_list'] = images_list
#     f['labels_list'] = labels_list
#
#
#     f.close()




''' pack into many files.h5'''

# shuffle
num_trainval = len(labels_trainval)
idx_rand = np.random.permutation(num_trainval)
images_trainval = [images_trainval[id] for id in idx_rand]
labels_trainval = [labels_trainval[id] for id in idx_rand]

num_test = len(labels_test)
idx_rand = np.random.permutation(num_test)
images_test = [images_test[id] for id in idx_rand]
labels_test = [labels_test[id] for id in idx_rand]

package_path = 'D:/eye/data/data_packages_crop_512'

batchsize = 128
width, height = 512, 512
images = []
labels = []
modes = ['trainval', 'test']

for mode in modes:
    print(mode)
    if mode == 'trainval':
        num = num_trainval
        impath = imgpath_trainval
    else:
        num = num_test
        impath = imgpath_test
    # impath = str(impath).encode('utf8')

    # num = 1000
    for i in range(num):
        if i%batchsize == 0:
            t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print('%s %d/%d'%(t_c, i, num))
            f = h5py.File(os.path.join(package_path, '%s_%d.h5'%(mode, i//batchsize)), 'w')
            images = []
            labels = np.zeros(shape=(batchsize), dtype=np.int)
            imgs = np.zeros(shape=(batchsize, height, width, 3), dtype=np.uint8)
            idx = 0

        if mode == 'trainval':
            image = images_trainval[i]
            label = labels_trainval[i]
        else:
            image = images_test[i]
            label = labels_test[i]

        with Image.open(os.path.join(impath, image)) as img:
            image = str(image).encode('utf8')
            img = img.convert('RGB')
            # img = img.resize((width, height))
            img = np.array(img, dtype=np.uint8)
            # f[image] = img
            imgs[idx] = img
            labels[idx] = int(label)
            images.append(image)
            idx += 1


        if (i+1)%batchsize == 0 or (i+1) == num:
            f['images'] = images
            f['labels'] = labels
            imgs = imgs[:idx]
            f['imgs'] = imgs
            f.close()

f = h5py.File(os.path.join(package_path, 'meta.h5'), 'w')
f['images_trainval'] = [str(each).encode('utf-8') for each in images_trainval]
f['labels_trainval'] = labels_trainval
f['images_test'] = [str(each).encode('utf-8') for each in images_test]
f['labels_test'] = labels_test
f['batchsize'] = batchsize
f['width'] = width
f['height'] = height
f.close()
