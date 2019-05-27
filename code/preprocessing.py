import numpy as np
import os
import csv
import h5py
import PIL.Image as Image
import time

root_path = '../kaggle'
imgpath_trainval = os.path.join(root_path, 'train').replace('\\', '/')
imgpath_test = os.path.join(root_path, 'test').replace('\\', '/')
trainval_id_path = os.path.join(root_path, 'trainLabels.csv').replace('\\', '/')
test_id_path = os.path.join(root_path, 'retinopathy_solution.csv').replace('\\', '/')

threshold = 0.8
num_error = 0

for mode in ['train', 'test']:
    print(mode)

    target_path_crop = '../kaggle/'+mode+'_crop'
    target_path_crop_resize = '../kaggle/'+mode+'_crop_512'
    if mode == 'train':
        id_path = trainval_id_path
        folder = imgpath_trainval
    else:
        id_path = test_id_path
        folder = imgpath_test

    files = os.listdir(folder)

    # with open(id_path, "r") as csvFile:
    for i, image in enumerate(files):
        if image[-5:] != '.jpeg':
            print('%s is not a jpeg'%image)
            continue
        # if i == 10:
        #     break

        # if i < 4586:
        #     continue

        if i%100 == 0:
            t_c = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print('%s %d/%d'%(t_c, i, len(files)))



        img = Image.open(os.path.join(folder, image))
        # img.show()
        img = img.convert('RGB')
        # img.show()
        img = np.array(img, dtype=np.uint8)
        # img = Image.fromarray(img)
        # img.save('d:/test/1.jpg')
        # img = np.array(img, dtype=np.uint8)
        # smooth, binary, center, radius
        # window = 3


        try:

            threshold_binary = 10
            threshold_boundary = 10
            # threshold_diameter = 1.2
            row, col, cha = img.shape
            # print(image)
            # print(img.shape)
            img_new = np.zeros(shape=[row, col], dtype=np.uint8)

            # pos = np.where(np.mean(img, axis=-1)<threshold_binary)
            # img_new[pos] = 0
            pos = np.where(np.mean(img, axis=-1) >= threshold_binary)
            img_new[pos] = 1
            # img_new1 = img_new
            # img_new1[pos] = 255
            # img_new1 = Image.fromarray(np.tile(img_new1[:,:,np.newaxis], (1,1,3)))
            # img_new1.save('d:/test/2.jpg')

            temp = np.sum(img_new, axis=0)
            pos = np.where(temp>threshold_boundary)
            left = np.min(pos)
            right = np.max(pos)
            left = max(left-int((right-left)*0.025), 0)
            right = min(right+int((right-left)*0.025), col-1)

            temp = np.sum(img_new, axis=1)
            pos = np.where(temp > threshold_boundary)
            up = np.min(pos)
            down = np.max(pos)
            up = max(up-int((down-up)*0.025), 0)
            down = min(down+int((down-up)*0.025), row-1)
            w_real = right-left+1
            h_real = down-up+1


            w = max(w_real, h_real)+1
            h = max(w_real, h_real)+1
            img_new = np.zeros(shape=(h, w, 3), dtype=np.uint8)
            shift_h = int((h-h_real)/2)
            shift_w = int((w-w_real)/2)

            img_new[shift_h:h_real+shift_h, shift_w:w_real+shift_w] = img[up:down+1, left:right+1, :]

        except Exception:
            print('%s image error'%image)
            img_new = img

        # print(img_new.shape)
        img_new = Image.fromarray(img_new)

        img_new.save(os.path.join(target_path_crop, image))

        img_new = img_new.resize(size=(512,512))
        img_new.save(os.path.join(target_path_crop_resize, image))

            # time.sleep(5)

