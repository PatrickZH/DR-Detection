import os
import math
import numpy as np
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import Linear
import PIL.Image as Image
import torch
from torch.utils.data.sampler import Sampler
import csv
import h5py
import copy

# no crop: mean, std
# im_mean = np.array([0.41044564, 0.28347273, 0.20089086], dtype=np.float)
# im_std = np.array([0.28900356, 0.20903871, 0.17090545], dtype=np.float)

# crop: mean, std
# im_mean = np.array([0.3460425078848989, 0.24051744191557664, 0.17183285806797155], dtype=np.float)
# im_std = np.array([0.2975885854228362, 0.21424068839221927, 0.1711617604127343], dtype=np.float)

step_trainval = 1
step_test = 1

shape_img = (512, 512)  # height width
im_mean = np.array([0.3460425078848989, 0.24051744191557664, 0.17183285806797155], dtype=np.float)
im_std = np.array([0.2975885854228362, 0.21424068839221927, 0.1711617604127343], dtype=np.float)
transform_train = transforms.Compose(
    [  # transforms.Resize(size=(int(shape_img[0]), int(shape_img[1]))),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
        #transforms.RandomGrayscale(1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomResizedCrop(size=shape_img[0], scale=(0.6, 1.0)),
        # transforms.RandomCrop(size=shape_img, padding=int(shape_img[0]/8)),
        transforms.ToTensor(), transforms.Normalize(im_mean, im_std)])
transform_test = transforms.Compose(
    [transforms.Resize(size=shape_img), transforms.ToTensor(), transforms.Normalize(im_mean, im_std)])
np.random.seed(20190507)



def data_process(dataset):
    if dataset == 'kaggle':
        root_path = '../'
        imgpath_trainval = os.path.join(root_path, 'nettraindata').replace('\\', '/')
        imgpath_test = os.path.join(root_path, 'nettestdata').replace('\\', '/') 
        trainval_id_path = os.path.join(root_path, 'nettraindata\\nettrainLabels.csv').replace('\\', '/')
        test_id_path = os.path.join(root_path, 'nettestdata\\testLabels.csv').replace('\\', '/')

        print('use data in %s, %s'%(imgpath_trainval, imgpath_test))
        print('shape_img', shape_img)

        threshold = 0.8
        num_error = 0
        # ratio_databalance = [1, 10, 5, 30, 32] # basic
        ratio_databalance = [1, 10, 5, 30, 30]
        files_trainval = os.listdir(imgpath_trainval)
        images_train = []
        labels_train = []
        images_val = []
        labels_val = []
        with open(trainval_id_path, "r") as csvFile:
            dict_reader = csv.DictReader(csvFile)
            for i, each in enumerate(dict_reader):
                # if i == 1000:
                #     break
                if np.random.rand()<0.3:
                    image = each['image']
                    level = int(each['level'])
                    repeat = ratio_databalance[level]
                    if image in files_trainval:
                        if np.random.rand()<threshold:
                            ##### data balance
                            for re in range(repeat):
                                images_train.append(os.path.join(imgpath_trainval, image).replace('\\', '/'))
                                labels_train.append(level)
                        else:
                            ##### data balance
                            for re in range(repeat):
                                images_val.append(os.path.join(imgpath_trainval, image).replace('\\', '/'))
                                labels_val.append(level)
                    else:
                        num_error +=1
        print('errors: %d wrong images in trainval data'%num_error)

        num_error = 0
        files_test = os.listdir(imgpath_test)
        images_test = []
        labels_test = []
        with open(test_id_path, "r") as csvFile:
            dict_reader = csv.DictReader(csvFile)
            for i, each in enumerate(dict_reader):
                # if i == 1000:
                #     break
                image = each['image'] + '.jpeg'
                level = int(each['level'])
                if image in files_test:
                    images_test.append(os.path.join(imgpath_test, image).replace('\\', '/'))
                    labels_test.append(level)
                else:
                    num_error +=1
        print('errors: %d wrong images in test data'%num_error)


        # shuffle
        idx_rand = np.random.permutation(len(labels_train))
        images_train = [images_train[id] for id in idx_rand]
        labels_train = [labels_train[id] for id in idx_rand]

        idx_rand = np.random.permutation(len(labels_val))
        images_val = [images_val[id] for id in idx_rand]
        labels_val = [labels_val[id] for id in idx_rand]

        idx_rand = np.random.permutation(len(labels_test))
        images_test = [images_test[id] for id in idx_rand]
        labels_test = [labels_test[id] for id in idx_rand]


        # for fast exp
        images_train = images_train[::step_trainval]
        labels_train = labels_train[::step_trainval]
        images_val = images_val[::step_trainval]
        labels_val = labels_val[::step_trainval]

        images_test = images_test[::step_test]
        labels_test = labels_test[::step_test]

        print('step_trainval: %d, step_test: %d'%(step_trainval, step_test))


        print('distribution')
        print('train:\t 0: %.2f,\t 1: %.2f,\t 2: %.2f,\t 3: %.2f,\t 4: %.2f'%(
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 0))[-1] / (len(labels_train)+1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 1))[-1] / (len(labels_train)+1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 2))[-1] / (len(labels_train)+1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 3))[-1] / (len(labels_train)+1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 4))[-1] / (len(labels_train)+1),))
        print('val:\t 0: %.2f,\t 1: %.2f,\t 2: %.2f,\t 3: %.2f,\t 4: %.2f'%(
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 0))[-1] / (len(labels_val)+1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 1))[-1] / (len(labels_val)+1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 2))[-1] / (len(labels_val)+1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 3))[-1] / (len(labels_val)+1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 4))[-1] / (len(labels_val)+1),))
        print('test:\t 0: %.2f,\t 1: %.2f,\t 2: %.2f,\t 3: %.2f,\t 4: %.2f'%(
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 0))[-1] / (len(labels_test)+1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 1))[-1] / (len(labels_test)+1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 2))[-1] / (len(labels_test)+1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 3))[-1] / (len(labels_test)+1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 4))[-1] / (len(labels_test)+1),))

        return images_train, labels_train, images_val, labels_val, images_test, labels_test, transform_train, transform_test


class dataset_construct(Dataset):
    def __init__(self, img_list, lab_list, transform=None):
        self.img_list = img_list
        self.lab_list = lab_list
        self.transform = transform

    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        name = self.img_list[idx]
        lab = self.lab_list[idx]
        img_path = name.replace('\\', '/')
        try:
            img = Image.open(img_path)
        except:
            print('error in loading image: %s'%img)
        if self.transform:
            degree = [0,90,180,270]
            rot_index = int(np.random.rand()*4)
            img = img.rotate(degree[rot_index])
            img = self.transform(img)
        return img, lab







def data_process_simple(dataset):
    if dataset == 'kaggle':
        root_path = '../data'

        f = h5py.File(os.path.join(root_path, 'meta.h5'), 'r')
        images_trainval = f['images_trainval'][:]
        images_trainval = [str(image, encoding = 'utf-8') for image in images_trainval]
        labels_trainval = f['labels_trainval'][:]
        images_test = f['images_test'][:]
        images_test = [str(image, encoding = 'utf-8') for image in images_test]
        labels_test = f['labels_test'][:]

        threshold = 0.8
        num_error = 0
        # ratio_databalance = [1, 10, 5, 30, 32] # basic
        ratio_databalance = [1, 10, 5, 30, 30]
        images_train = []
        labels_train = []
        images_val = []
        labels_val = []

        for i, image in enumerate(images_trainval):
            level = labels_trainval[i]
            repeat = ratio_databalance[level]
            if np.random.rand()<threshold:
                ##### data balance
                for re in range(repeat):
                    images_train.append(image)
                    labels_train.append(level)
            else:
                ##### data balance
                for re in range(repeat):
                    images_val.append(image)
                    labels_val.append(level)



        # shuffle
        idx_rand = np.random.permutation(len(labels_train))
        images_train = [images_train[id] for id in idx_rand]
        labels_train = [labels_train[id] for id in idx_rand]

        idx_rand = np.random.permutation(len(labels_val))
        images_val = [images_val[id] for id in idx_rand]
        labels_val = [labels_val[id] for id in idx_rand]

        idx_rand = np.random.permutation(len(labels_test))
        images_test = [images_test[id] for id in idx_rand]
        labels_test = [labels_test[id] for id in idx_rand]



        # for fast exp
        images_train = images_train[::step_trainval]
        labels_train = labels_train[::step_trainval]
        images_val = images_val[::step_trainval]
        labels_val = labels_val[::step_trainval]

        images_test = images_test[::step_test]
        labels_test = labels_test[::step_test]

        print('step_trainval: %d, step_test: %d'%(step_trainval, step_test))

        print('distribution')
        print('train:\t 0: %.2f,\t 1: %.2f,\t 2: %.2f,\t 3: %.2f,\t 4: %.2f' % (
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 0))[-1] / (len(labels_train) + 1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 1))[-1] / (len(labels_train) + 1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 2))[-1] / (len(labels_train) + 1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 3))[-1] / (len(labels_train) + 1),
            np.shape(np.where(np.asarray(labels_train, dtype=int) == 4))[-1] / (len(labels_train) + 1),))
        print('val:\t 0: %.2f,\t 1: %.2f,\t 2: %.2f,\t 3: %.2f,\t 4: %.2f' % (
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 0))[-1] / (len(labels_val) + 1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 1))[-1] / (len(labels_val) + 1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 2))[-1] / (len(labels_val) + 1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 3))[-1] / (len(labels_val) + 1),
            np.shape(np.where(np.asarray(labels_val, dtype=int) == 4))[-1] / (len(labels_val) + 1),))
        print('test:\t 0: %.2f,\t 1: %.2f,\t 2: %.2f,\t 3: %.2f,\t 4: %.2f' % (
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 0))[-1] / (len(labels_test) + 1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 1))[-1] / (len(labels_test) + 1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 2))[-1] / (len(labels_test) + 1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 3))[-1] / (len(labels_test) + 1),
            np.shape(np.where(np.asarray(labels_test, dtype=int) == 4))[-1] / (len(labels_test) + 1),))

        return images_train, labels_train, images_val, labels_val, images_test, labels_test, transform_train, transform_test



class dataset_construct_from_memory(Dataset):
    def __init__(self, img_list, lab_list, mode, transform=None):
        if mode == 'train' or mode == 'val':
            data_path = '../data/data_512_trainval_crop.h5'
        else:
            data_path = '../data/data_512_test_crop.h5'
        print('use data in %s'%(data_path))
        print('shape_img', shape_img)

        f = h5py.File(data_path, 'r')
        # images_list = f['images_list'][:]
        # labels_list = f['labels_list'][:]

        self.images = []
        for img in img_list:
            name = str(str(img).split('/')[-1]).encode('utf-8')
            self.images.append(f[name][:])

        self.img_list = img_list
        self.lab_list = lab_list
        self.transform = transform


        # f.close()
        print('data construct: %s %d'%(mode, len(self.lab_list)))

    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        name = str(str(self.img_list[idx]).split('/')[-1]).encode('utf-8')
        lab = self.lab_list[idx]
        # try:
        img = self.images[idx]
        # img = self.f[name][:]

        # except:
        #     print('error in loading image: %s' % name)
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, lab








def data_process_batches(dataset):
    if dataset == 'kaggle':
        root_path = '../data/data_packages_crop_512'
        threshold = 0.8

        files = os.listdir(root_path)
        batches_train, batches_val, batches_test = [], [], []
        for f in files:
            if len(f)>8 and f[:8] == 'trainval':
                if np.random.rand() < threshold:
                    batches_train.append(os.path.join(root_path, f))
                else:
                    batches_val.append(os.path.join(root_path, f))
            if len(f)>4 and f[:4] == 'test':
                batches_test.append(os.path.join(root_path, f))

        return batches_train, batches_val, batches_test, transform_train, transform_test



class dataset_construct_from_batches(Dataset):
    def __init__(self, batches_list, mode, transform=None):
        tokens = str(batches_list[0]).split('_')
        s = ''
        for t in tokens[:-1]:
            s += t
            s += '_'
        print('use data in batches %s'%(s))
        print('shape_img', shape_img)

        self.batches_list = batches_list
        self.transform = transform

        print('data construct from batches: %s %d'%(mode, len(self.batches_list)))

    def __len__(self):
        return len(self.batches_list)

    def __getitem__(self, idx):
        batch_path = self.batches_list[idx]
        f = h5py.File(batch_path, 'r')
        imgs = f['imgs'][:]
        labels = f['labels'][:]
        images = f['images'][:]
        shape = imgs.shape
        imgs_t = np.zeros(shape=(shape[0], shape[3], shape[1], shape[2]))

        if self.transform:
            for i, img in enumerate(imgs):
                img = Image.fromarray(img)
                img = self.transform(img)
                imgs_t[i] = img
        return imgs_t, labels






def one_hot(y, c):
    """
        Parameters
        ----------
        y: the original labels, size = (n)
        c: the total number of classes

        Returns
        ----------
        y_onehot: the one-hot labels
    """
    y = np.array(y, dtype=np.int)
    n = np.shape(y)[0]
    y_onehot = np.zeros(shape=(n, c),dtype=np.int)
    for i in range(n):
        y_onehot[i, y[i]] = 1
    return y_onehot


# def distance(x, y, part = 100, cuda = False):
#     n, d = np.shape(x)[0], np.shape(x)[1]
#     m, d = np.shape(y)[0], np.shape(y)[1]
#     dist = np.zeros(shape=(n, m))
#     steps = math.ceil(m/part)
#     for i in range(steps):
#         # if steps>1:
#             # t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
#             # print('%s %d/%d'%(t_c, i, steps))
#         idx = np.arange(i*part, min((i+1)*part, m))
#         x_ = x[:, np.newaxis, :]
#         x_ = np.tile(x_, (1, len(idx), 1))
#         y_ = y[idx]
#         y_ = y_[np.newaxis, :, :]
#         y_ = np.tile(y_, (n, 1, 1))
#         if cuda:
#             dist_ = torch.norm(torch.FloatTensor(x_).cuda() - torch.FloatTensor(y_).cuda(), dim=-1)
#             dist_ = dist_.cpu().data.numpy()
#         else:
#             dist_ = np.linalg.norm(x_ - y_, axis=-1, keepdims=False)
#         dist[:, idx] = dist_
#
#
#     # x = x[:, np.newaxis, :]
#     # x = np.tile(x, (1, m, 1))
#     # y = y[np.newaxis, :, :]
#     # y = np.tile(y, (n, 1, 1))
#     # dist = np.linalg.norm(x - y, axis=-1, keepdims=False)
#     return dist
