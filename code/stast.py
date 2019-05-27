import os

import csv
import os
import numpy as np

root_path = '../kaggle'
imgpath_trainval = os.path.join(root_path, 'train').replace('\\', '/')
imgpath_test = os.path.join(root_path, 'test').replace('\\', '/')
annotationpath = os.path.join(root_path, 'trainLabels.csv').replace('\\', '/')
test_id_path = os.path.join(root_path, 'retinopathy_solution.csv').replace('\\', '/')

threshold = 0.8
num_error = 0
files_trainval = os.listdir(imgpath_trainval)
images_train = []
labels_train = []
images_val = []
labels_val = []
with open(annotationpath, "r") as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for each in dict_reader:
        image = each['image'] + '.jpeg'
        level = int(each['level'])
        if image in files_trainval:
            if np.random.rand() < threshold:
                images_train.append(image)
                labels_train.append(level)
            else:
                images_val.append(image)
                labels_val.append(level)
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

print('labels_train + labels_val')
base = 0
for level in range(max(labels_train+labels_val)+1):
    num = 0
    for lab in labels_train+labels_val:
        if lab == level:
            num += 1
    if level == 0:
        base = num
    print('level %d, num = %d, ratio = %.1f'%(level, num, base/num))


# print('labels_val')
# for level in range(max(labels_val)):
#     num = 0
#     for lab in labels_val:
#         if lab == level:
#             num += 1
#     print('level %d, num = %d'%(level, num))

print('labels_test')
for level in range(max(labels_test)+1):
    num = 0
    for lab in labels_test:
        if lab == level:
            num += 1
    if level == 0:
        base = num
    print('level %d, num = %d, ratio = %.1f'%(level, num, base/num))

print('x')