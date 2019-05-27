import numpy as np
import os
import csv
import h5py
import PIL.Image as Image
import torchvision.transforms as transforms

from multiprocessing.dummy import Pool as ThreadPool
import time




root_path = '../kaggle'
imgpath_trainval = os.path.join(root_path, 'train_crop_512').replace('\\', '/')
imgpath_test = os.path.join(root_path, 'test_crop_512').replace('\\', '/')
trainval_id_path = os.path.join(root_path, 'trainLabels.csv').replace('\\', '/')
test_id_path = os.path.join(root_path, 'retinopathy_solution.csv').replace('\\', '/')

threshold = 0.8
num_error = 0
# ratio_databalance = [1, 10, 5, 30, 32] # basic
# ratio_databalance = [1, 10, 5, 10, 10]
files_trainval = os.listdir(imgpath_trainval)
images_trainval = []
labels_trainval = []
images_all = np.zeros(shape=(40000, 128, 128, 3))
transform_test = transforms.Compose([transforms.ToTensor()])


# temp_list = []


t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
print('%s begin'%t_c)

with open(trainval_id_path, "r") as csvFile:
    dict_reader = csv.DictReader(csvFile)
    for i, each in enumerate(dict_reader):
        if i % 5000==0:
            # print(i)
            break
        images_all[i] = np.random.rand(128, 128, 3)
        image = each['image'] + '.jpeg'
        level = int(each['level'])
        # repeat = ratio_databalance[level]
        if image in files_trainval:
            # temp_list.append(image)
            img = Image.open(os.path.join(imgpath_trainval, image))
            # img.show()
            img = img.convert('RGB')
            # img.show()
            img = img.resize((128, 128))
            # img.show()
            img = transform_test(img)
            img = np.array(img)
            # print(np.max(img))
            # img = Image.fromarray(img)
            # img.show()
            # img = img/255
            images_all[i] = np.transpose(img,axes=[1, 2, 0])





images_all = images_all[:i]
print('mean: ', np.mean(images_all[:,:,:,0]), np.mean(images_all[:,:,:,1]), np.mean(images_all[:,:,:,2]))
print('std: ', np.std(images_all[:,:,:,0]), np.std(images_all[:,:,:,1]), np.std(images_all[:,:,:,2]))

t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
print('%s finish' % t_c)


