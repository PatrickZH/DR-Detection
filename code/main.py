import pickle
import numpy as np
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from utils import data_process, data_process_simple, data_process_batches, dataset_construct, dataset_construct_from_memory, dataset_construct_from_batches, one_hot
import net_resnet


def continuous_kappa(y, t, y_pow=1, eps=1e-15):
    # if y.ndim == 1:
    y = one_hot(y, c=5)

    # if t.ndim == 1:
    t = one_hot(t, c=5)

    # Weights.
    num_scored_items, num_ratings = y.shape
    ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                          reps=(1, num_ratings))
    ratings_squared = (ratings_mat - ratings_mat.T) ** 2
    weights = ratings_squared / float(num_ratings - 1) ** 2

    if y_pow != 1:
        y_ = y ** y_pow
        y_norm = y_ / (eps + y_.sum(axis=1)[:, None])
        y = y_norm

    hist_rater_a = np.sum(y, axis=0)
    hist_rater_b = np.sum(t, axis=0)

    conf_mat = np.dot(y.T, t)

    nom = weights * conf_mat
    denom = (weights * np.dot(hist_rater_a[:, None],
                              hist_rater_b[None, :]) /
             num_scored_items)

    return 1 - nom.sum() / denom.sum()
        #, conf_mat, hist_rater_a, hist_rater_b, nom, denom


def epoch(mode, Iter, dataloader, batchsize, net, criterion_MSE, criterion_CE, alpha = 1, lr=0.001, decay=0.0005):
    # train
    loss_avg = 0
    loss_CE_avg = 0
    loss_MSE_avg = 0
    acc_avg = 0
    num_exp = 0
    tstart = time.clock()
    num_gpu = torch.cuda.device_count()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay)

    len_dl = len(dataloader)
    ans_all = []
    pred_all = []
    imgs_pool = None
    labs_pool = None
    size_pool = 0

    for i_batch, datum in enumerate(dataloader):
        if i_batch in range(0, len_dl, len_dl // 10 +1):
            print('%d/%d' % (i_batch, len_dl), end=' ')

        # t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
        # print('begin', t_c)

        img = datum[0]
        lab = datum[1]


        # shape = img.shape
        # img = img.float()
        # if len(shape)>4:
        #     img.resize_(shape[0]*shape[1], shape[2], shape[3], shape[4])
        #     lab.resize_(shape[0]*shape[1])
        #     n_b = shape[0]*shape[1]
        # else:
        #     n_b = np.shape(lab)[0]
        #
        # if size_pool > 0:
        #     imgs_pool = torch.cat((imgs_pool, img), dim=0)
        #     labs_pool = torch.cat((labs_pool, lab), dim=0)
        # else:
        #     imgs_pool = img
        #     labs_pool = lab
        #
        # size_pool += n_b
        #
        # while size_pool >= batchsize or (size_pool>=num_gpu*2 and i_batch==len_dl-1):
        #     img = imgs_pool[:min(batchsize, size_pool)]
        #     lab = labs_pool[:min(batchsize, size_pool)]
        #     size_pool -= batchsize
        #     if size_pool>0:
        #         imgs_pool = imgs_pool[batchsize:]
        #         labs_pool = labs_pool[batchsize:]


        img = img.float()
        img = img.cuda()
        n_b = np.shape(lab)[0]
        lab = lab.long()
        lab = lab.cuda()


        if mode == 'train':
            net.train()
            if n_b < num_gpu*2:
              continue
        else:
            net.eval()


        # classification loss
        output, feat = net(img)
        indices = torch.tensor([0, 1, 2, 3, 4]).cuda()
        logist = torch.index_select(output, -1, indices)
        indices = torch.tensor([5]).cuda()
        value = torch.index_select(output, -1, indices)

        # regress loss
        lab_f = lab.float()
        lab_f = lab_f.cuda()
        loss_MSE = criterion_MSE(value, lab_f+0.5)
        loss_CE = criterion_CE(logist, lab)
        acc_CE = np.sum(np.equal(np.argmax(logist.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss = alpha*loss_MSE + loss_CE
        acc = acc_CE

        loss_avg += loss.item()
        loss_CE_avg += loss_CE.item()
        loss_MSE_avg += loss_MSE.item()
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        pred_all = pred_all + np.argmax(logist.cpu().data.numpy(), axis=-1).tolist()
        ans_all = ans_all + lab.cpu().data.numpy().tolist()

        # t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
        # print('end', t_c)


    loss_avg /= num_exp
    loss_CE_avg /= num_exp
    loss_MSE_avg /= num_exp
    acc_avg /= num_exp

    measure = continuous_kappa(pred_all, ans_all, y_pow=1, eps=1e-15)

    tend = time.clock()
    tcost = tend - tstart
    print()
    t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('%s %s Iter: %d \t time = %.1f \t loss = %.6f \t loss_CE = %.6f \t loss_MSE = %.6f \t measure = %.2f \t acc = %.2f' % (t_c, mode, Iter, tcost, loss_avg, loss_CE_avg, loss_MSE_avg, measure, acc_avg))

    if mode == 'test' or mode == 'val' or mode == 'train':
        for level in range(5):
            pos = [p for p, an in enumerate(ans_all) if an == level]
            sensitivity = np.sum(np.equal(np.asarray([pred_all[p] for p in pos], dtype=int), level)) / len(pos)
            pos = [p for p, an in enumerate(ans_all) if an != level]
            specificity = 1 - np.sum(np.equal(np.asarray([pred_all[p] for p in pos], dtype=int), level)) / len(pos)
            print('level = %d, sensitivity = %.1f, specificity = %.1f' % (level, sensitivity, specificity))


def main():
    # print('mark: with strong data augmentation')
    dataset = 'kaggle'
    alpha = 0
    batchsize = 256
    worknum = 8
    lr = 0.001
    gamma = 0.5
    # schedules = [81, 122, 200]
    iteration = 200
    schedules = range(50, iteration, 25)
    momentum = 0.9
    decay = 0.0005
    num_class = 5
    # num_metric = 1
    network_depth = 18
    dim = 256
    pretrain = True
    # metric = 'CEL'

    print('dataset', dataset)
    print('alpha', alpha)
    print('batchsize', batchsize)
    print('worknum', worknum)
    print('lr', lr)
    print('gamma', gamma)
    print('schedules', schedules)
    print('decay', decay)
    print('iteration', iteration)
    print('num_class', num_class)
    print('network_depth', network_depth)
    print('dim', dim)
    print('pretrain', pretrain)


    num_output = num_class + 1

    # ''' load from batches'''
    # batches_train, batches_val, batches_test, transform_train, transform_test = data_process_batches(dataset)
    # dataset_train = dataset_construct_from_batches(batches_train, mode='train', transform=transform_train)
    # dataset_val = dataset_construct_from_batches(batches_val, mode='val', transform=transform_test)
    # dataset_test = dataset_construct_from_batches(batches_test, mode='test', transform=transform_test)
    #
    # dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    # dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    # # dataloader_test = DataLoader(dataset_test, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)




    ''' load from memory'''
    # images_train, labels_train, images_val, labels_val, images_test, labels_test, transform_train, transform_test = data_process_simple(dataset)
    # dataset_train = dataset_construct_from_memory(images_train, labels_train, mode='train', transform=transform_train)
    # dataset_val = dataset_construct_from_memory(images_val, labels_val, mode='val', transform=transform_test)
    # # dataset_test = dataset_construct_from_memory(images_test, labels_test, mode='test', transform=transform_test)
    #
    # dataloader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=worknum,
    #                               pin_memory=True)
    # dataloader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=True, num_workers=worknum, pin_memory=True)
    # # dataloader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=True, num_workers=worknum, pin_memory=True)



    ''' load from images'''
    images_train, labels_train, images_val, labels_val, images_test, labels_test, transform_train, transform_test = data_process(dataset)
    dataset_train = dataset_construct(images_train, labels_train, transform=transform_train)
    dataset_val = dataset_construct(images_val, labels_val, transform=transform_test)
    # dataset_test = dataset_construct(images_test, labels_test, transform=transform_test)

    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, shuffle=True, num_workers=worknum,
                                  pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batchsize, shuffle=True, num_workers=worknum, pin_memory=True)
    # dataloader_test = DataLoader(dataset_test, batch_size=batchsize, shuffle=True, num_workers=worknum, pin_memory=True)

    num_train = len(dataset_train)
    num_val = len(dataset_val)
    num_test = 0 #len(dataset_test)
    print('num_train = ', num_train, 'num_val = ', num_val, 'num_test = ', num_test)


    t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
    print('%s finish data load'%t_c)



    # train
    if network_depth == 18:
        net = net_resnet.resnet18(num_features=dim, num_classes=num_output, pretrained=pretrain).cuda()
    elif network_depth == 50:
        net = net_resnet.resnet50(num_features=dim, num_classes=num_output, pretrained=pretrain).cuda()
    else:
        net = None
        print('error: unknown network structure')
        exit()


    # GPU
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        print('GPU number = %d' % (num_gpu))
        device_ids = np.arange(num_gpu).tolist()
        print('device_ids:')
        print(device_ids)
        net = net.cuda()
        net = nn.DataParallel(net, device_ids=device_ids).cuda()
    else:
        print('only cpu is available')


    criterion_MSE = nn.MSELoss().cuda()
    criterion_CE = nn.CrossEntropyLoss().cuda()

    print('lr = %f' % (lr))





    for Iter in range(iteration):

        # learning rate
        if Iter in schedules:
            lr *= gamma
            print('lr = %f'%(lr))


        epoch('train', Iter, dataloader_train, batchsize, net, criterion_MSE, criterion_CE, alpha=alpha, lr=lr, decay=decay)


        # if Iter%3 == 0:
        epoch('val', Iter, dataloader_val, batchsize, net, criterion_MSE, criterion_CE, alpha=alpha)
            # epoch('test', Iter, dataloader_test, batchsize, net, criterion_MSE, criterion_CE, alpha=alpha)



    print('xxx')
if __name__ == '__main__':
    main()
