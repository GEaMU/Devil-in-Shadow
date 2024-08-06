
from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
from network.lightcnn import LightCNN_29v2
from data.dataset import Dataset
from sklearn.metrics.pairwise import cosine_similarity
# Use default value
parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--model', default='LightCNN-9', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--num_classes', default=725, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')

## input if necessary
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='root path of face images (default: none).')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--img_list', default='', type=str, metavar='PATH',
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--protocols', default='protocols', type=str, metavar='PATH',
                    help='list of protocols (default: none).')
# parser.add_argument('--save_path', default='', type=str, metavar='PATH',
#                     help='save root path for features of face images.')

postfix = None


def main():
    stat = []
    '''
    for i in range(2, 10):
        global postfix
        postfix = i
        print()
        print('---------------------------------')
        print('Iteration: ', postfix-1)
        avg_r_a, std_r_a, avg_v_a, std_v_a = excute()
        # print(avg_rank1_acc, avg_vr_acc)
        stat.append([avg_r_a, std_r_a, avg_v_a, std_v_a])
        # exit()
    '''
    # global postfix  #sgong
    # postfix = ""  #sgong
    print()
    print('---------------------------------')
    # print('Iteration: ', postfix-1)   #sgong
    avg_r_a, std_r_a, avg_v_a, std_v_a = excute()
    # print(avg_rank1_acc, avg_vr_acc)
    stat.append([avg_r_a, std_r_a, avg_v_a, std_v_a])
    print(stat)
    exit()


def excute():
    global args
    args = parser.parse_args()

    # print(args)
    # exit()
    if args.root_path == '':
        #args.root_path = '/media/zli33/DATA/study/AdvCompVision/Project/Implementation/mtcnn-pytorch-master/NIR-VIS-2.0'
        args.root_path = '/data/MTCNN_Portable/CASIA_O_MTCNN'#'/data/HFR Datasets 2022-10/CASIA NIR-VIS 2.0 Database/NIR-VIS 2.0 Database/原文件/NIR-VIS-2.0'#'/data/HFGI-main/CASIA_DLIB'
        #o_root_path='/data/HFR Datasets 2022-10/CASIA NIR-VIS 2.0 Database/NIR-VIS 2.0 Database/原文件/NIR-VIS-2.0'
    if args.resume == '':
        args.resume ='./pre_train/LightCNN_29Layers_V2_checkpoint.pth'# './trained_model/LightCNN128_epoch_15.pth.tar'#'./pre_train/LightCNN_29Layers_V2_checkpoint.pth'#
    if args.protocols == '':
        args.protocols = 'protocols'
    newroot_path='/data/DPR-master/CASIA_mtcnn_re'#'/data/MTCNN_Portable/CASIA_O_MTCNN'#'/data/DVG2/DVG-Face-master/attack_CASIA'#'/data/PSFRGAN-master/CASIA_hQ'#'/data/MTCNN_Portable/CASIA_O_MTCNN'#''/data/DPR-master/CASIA_relight_128'#'/data/HFGI-main/CASIA_DLIB'#'/data/DVG2/DVG-Face-master/attack_CASIA'#'/data/pytorch-CycleGAN-and-pix2pix-master/results/CASIA128'#'/data/DPR-master/CASIA_relight'#
    ##'/data/MTCNN_Portable/CASIA_O_MTCNN'#
    model = LightCNN_29v2(num_classes=args.num_classes)
    load_model(model, "./model/LightCNN128_epoch_15.pth.tar")##./pre_train/LightCNN_29Layers_V2_checkpoint.pth
    model.eval()
    # train_loader = torch.utils.data.DataLoader(
    #     Dataset(args), batch_size=args.batch_size, shuffle=None, num_workers=args.workers, pin_memory=True)
    # gallery_file_list = 'vis_gallery_*.txt'
    # probe_file_list = 'nir_probe_*.txt'
    # import glob2
    # gallery_file_list = glob2.glob(args.root_path + '/' + args.protocols + '/' + gallery_file_list)
    # probe_file_list = glob2.glob(args.root_path + '/' + args.protocols + '/' + probe_file_list)
    # # remove *_dev.txt file in both list
    # gallery_file_list = sorted(gallery_file_list)[0:-1]
    # probe_file_list = sorted(probe_file_list)[0:-1]
    gallery_file_list=[]
    probe_file_list=[]
    gallery_file = '/data/HFR Datasets 2022-10/CASIA NIR-VIS 2.0 Database/NIR-VIS 2.0 Database/原文件/NIR-VIS-2.0/protocols/vis_gallery_1.txt'#'/data/DPR-master/CASIA_relight/vis_train_time.txt'
    probe_file ='/data/DPR-master/CASIA_relight/relight_probe_256time.txt'# 'nir_probe_1.txt'# '/data/HFGI-main/CASIA_DLIB/myprotocols.txt'#'nir_probe_1.txt'#
    gallery_file_list.append(gallery_file)
    probe_file_list.append(probe_file)

    avg_r_a, std_r_a, avg_v_a, std_v_a = load(model, args.root_path, gallery_file_list, probe_file_list,newroot_path,args.protocols,)
    return avg_r_a, std_r_a, avg_v_a, std_v_a


def load(model, root_path, gallery_file_list, probe_file_list,newroot_path,protocols):
    cosine_similarity_score = []
    rank1_acc = []
    vr_acc = []
    g_count = 0
    # read sub_experiment from gallery file list 0-10
    for i in range(len(gallery_file_list)):
        g_count += 1

        transform = transforms.Compose([
            #transforms.CenterCrop(128),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        count = 0
        input = torch.zeros(1, 1, 128, 128)
        input0=torch.zeros(1, 1, 128, 128)
        ccount = 0
        simi_dict = {}
        probe_features = []
        gallery_features = []
        probe_names = []
        gallery_names = []
        probe_img_name_list = []
        gallery_img_name_list = []

        # read images in each sub_experiment
        # in the first time read images from probe image file 1
        print('==> extract feature from the image')

        probe_img_list = read_list(probe_file_list[i])#os.path.join(newroot_path,protocols,probe_file_list[i])
        print('===> probe image list')
        for probe_img_name in probe_img_list:

            start = time.time()
            count = count + 1
            label = 0
            simi_dict[count] = []
            #probe_img_feature, folder_name = feature_extract(probe_img_name, input, transform, model, newroot_path)
            probe_img_feature, folder_name=relight_feature_extract(probe_img_name, input, transform, model, newroot_path, root_path, input0)
            # print(probe_img_feature.shape)
            # exit()

            # save_feature(root_path, probe_img_name, probe_img_feature)

            # exit()

            # features.data.cpu().numpy()[0]
            end = time.time() - start
            # print("{}({}/{}). Time: {:.4f}".format(os.path.join(root_path, probe_img_name), count, len(probe_img_list), end))
            # if count == 10:
            #     break

            # continue
            probe_features.append(probe_img_feature)
            probe_names.append(folder_name)
            probe_img_name_list.append(probe_img_name)

        gallery_img_list = read_list(os.path.join(root_path,protocols,gallery_file_list[i]))
        print('===> gallery image list')
        for gallery_img_name in gallery_img_list:
            # break
            #gallery_img_name = revise_name(gallery_img_name)
            # print("gallery_img_name:",gallery_img_name)
            start = time.time()
            ccount = ccount + 1
            gallery_img_feature, folder_name = feature_extract(gallery_img_name, input, transform, model, root_path)
            # print(gallery_img_feature.shape)
            # save_feature(root_path, gallery_img_name, gallery_img_feature)
            # exit()

            end = time.time() - start
            # print("{}({}/{}). Time: {:.4f}".format(os.path.join(root_path, gallery_img_name), ccount, len(gallery_img_list), end))
            # continue

            gallery_features.append(gallery_img_feature)
            gallery_names.append(folder_name)
            gallery_img_name_list.append(gallery_img_name)

        # continue

        probe_features = np.array(probe_features)

        gallery_features = np.array(gallery_features)

        # print('probe_features.shape=', probe_features.shape)
        # print('gallery_features.shape=', gallery_features.shape)
#-----------------------------------------------------------------------------------
        #cosine-distance


        score = cosine_similarity(gallery_features, probe_features).T
#-----------------------------------------------------------------------------------




        # print('score.shape= ', score.shape)
        # exit()
        r_acc, tpr = compute_metric(score, probe_names, gallery_names, g_count, probe_img_name_list,
                                    gallery_img_name_list)
        # print('score={}, probe_names={}, gallery_names={}'.format(score, probe_names, gallery_names))
        rank1_acc.append(r_acc)
        vr_acc.append(tpr)

    # print('over')
    # exit()

    avg_r_a = np.mean(np.array(rank1_acc))
    std_r_a = np.std(np.array(rank1_acc))
    avg_v_a = np.mean(np.array(vr_acc))
    std_v_a = np.std(np.array(vr_acc))
    # print(avg)
    # avg_rank1_acc = sum(rank1_acc)/(len(rank1_acc) + 1e-5)
    # avg_vr_acc = sum(vr_acc)/(len(vr_acc) + 1e-5)
    print()
    print('=====================================================')
    print('Final Rank1 accuracy is', avg_r_a * 100, "% +", std_r_a)
    print('Final VR@FAR=0.1% accuracy is', avg_v_a * 100, "% +", std_v_a)
    print('=====================================================')
    print()
    return avg_r_a, std_r_a, avg_v_a, std_v_a


def revise_name(probe_img_name):
    # print(probe_img_name, type(probe_img_name))
    suffix = probe_img_name.split('.')
    # if suffix[-1] != 'bmp':
    #     suffix[-1] = 'bmp'

    probe_img_name = '.'.join(suffix)
    revise_name = probe_img_name.split('\\')
    # print(revise_name)
    # # use '_128x128' when evaluate cropped image provided by dataset
    # revise_name[1] += '_128x128'
    # # use '_crop' when evaluate cropped image provided by zhenggang, scale = 48/100
    # # use '_80' when evaluate cropped image provided by zhenggang, scale = 48/80
    # # use '_80' when evaluate cropped image provided by zhenggang, scale = 48/110
    # global postfix   #sgong
    # revise_name[1] += '_train'
    # revise_name[1] += '_crop' + str(postfix)
    #revise_name[1] += '_128x128'  # sgong
    # print(revise_name)
    # exit()
    temp = ""
    for i in range(len(revise_name)):
        temp = temp + revise_name[i]
        if i != len(revise_name) - 1:
            temp += '\\'
            print(temp)
    return temp


def compute_metric(score, probe_names, gallery_names, g_count, probe_img_list, gallery_img_list, ):
    # print('score.shape =', score.shape)
    # print('probe_names =', np.array(probe_names).shape)
    # print('gallery_names =', np.array(gallery_names).shape)
    print('===> compute metrics')
    # print(probe_names[1], type(probe_names[1]))
    # exit()
    label = np.zeros_like(score)
    maxIndex = np.argmax(score, axis=1)
    # print('len = ', len(maxIndex))
    count = 0
    for i in range(len(maxIndex)):
        probe_names_repeat = np.repeat([probe_names[i]], len(gallery_names), axis=0).T
        # compare two string list
        result = np.equal(probe_names_repeat, gallery_names) * 1
        # result = np.core.defchararray.equal(probe_names_repeat, gallery_names) * 1
        # find the index of image in the gallery that has the same name as probe image
        # print(result)
        # print('++++++++++++++++++++++++++++++++=')
        index = np.nonzero(result == 1)

        # if i == 10:
        #     exit()
        if len(index[0]) != 1:
            print('more than one identity name in gallery is same as probe image name')
            ind = index[0]
        else:
            label[i][index[0][0]] = 1
            ind = index[0][0]

        # find the max similarty score in gallery has the same name as probe image
        if np.equal(int(probe_names[i]), int(gallery_names[maxIndex[i]])):
            count += 1
        else:
            pass
            # print(probe_img_list[i], gallery_img_list[ind])

        # flag = np.equal(probe_names[i], gallery_names[i])

        # labelOfmaxIndex.append(label[0, i])
    # print('count = ', count)
    r_acc = count / (len(probe_names) )#+ 1e-5

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(label.flatten(), score.flatten())

    print("In sub_experiment", g_count, 'count of true label :', count)
    print('rank1 accuracy =', r_acc)
    print('VR@FAR=0.1% accuracy =', tpr[fpr <= 0.001][-1])

    # plot_roc(fpr, tpr, thresholds, g_count)
    return r_acc, tpr[fpr <= 0.001][-1]

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def relight_feature_extract(img_name, input, transform, model, root_path,old_root_path,input0):
    # print(img_name)
    new_name=[]
    try:
        folder_name = img_name.split("\\")[-2]
        new = img_name.split("\\")[1]
        # print(new)
        # if new == 'NIR':
        #     new = 'NIR_128x128'
        # else:
        #
        #     new = 'VIS_128x128'
        # print(new)
        temp_name = img_name.split('\\')[3]
        temp = temp_name.split('.')
        # temp_name=temp_name.split('.')[-2]
        # print(temp_name)
        # if temp[-1] == 'jpg':
        #     temp[-1] = 'bmp'
        temp_name = '.'.join(temp)
        # print(temp_name)
        new_name.append(img_name.split('\\')[0])
        new_name.append(new)
        new_name.append(img_name.split('\\')[2])
        new_name.append(temp_name)
        img_name = '/'.join(new_name)
        #img_name = '/'.join(img_name.split('\\'))
    except:
        folder_name = img_name.split("/")[-2]
        new = img_name.split("/")[1]
        # print(new)

        temp_name = img_name.split('/')[3]
        temp = temp_name.split('.')
        temp_name = '.'.join(temp)
        new_name.append(img_name.split('/')[0])
        new_name.append(new)
        new_name.append(img_name.split('/')[2])
        new_name.append(temp_name)
        img_name = '/'.join(new_name)

    # print(folder_name)
    old_img_name = '/'.join(img_name.split('\\'))
    # print(img_name)
    # path = os.getcwd() + '/NIR-VIS-2.0/'
    path = root_path  # sgong
    #lable=torch.Tensor([folder_name])
    # print(os.path.join(path, img_name))

    #img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
    print(os.path.join(path, img_name))
    img=Image.open(os.path.join(path, img_name))
    old_img=Image.open(os.path.join(old_root_path, old_img_name))
    if img is None:
        # print('image not found')
        print(os.path.join(path, img_name))
        pp = os.path.join(path, img_name).split('\\')
        temp = pp[-1].split('.')
        print(temp)
        # if temp[-1] == 'bmp':
        #     temp[-1] = 'jpg'
        # elif temp[-1] == 'jpg':
        #     temp[-1] = 'bmp'
        temp = '.'.join(temp)
        print(temp)
        pp[-1] = temp
        i_p = '/'.join(pp)
        print('ip:'+i_p)
        img = Image.open(i_p)#cv2.imread(i_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('image not found')
            print(i_p)
            exit()
        # return 0




    with torch.no_grad():
        input_var = torch.autograd.Variable(input)
    feature = model(input_var)[1]
    print(folder_name)
    return feature.data.cpu().numpy()[0], int(folder_name)

def feature_extract(img_name, input, transform, model, root_path):
    # print(img_name)
    new_name=[]
    try:
        folder_name = img_name.split("\\")[-2]
        new = img_name.split("\\")[1]
        temp_name = img_name.split('\\')[3]
        temp = temp_name.split('.')#2 years ago
        temp_name = '.'.join(temp)
        # print(temp_name)
        new_name.append(img_name.split('\\')[0])
        new_name.append(new)
        new_name.append(img_name.split('\\')[2])
        new_name.append(temp_name)
        img_name = '/'.join(new_name)
    except:
        folder_name = img_name.split("/")[-2]
        new = img_name.split("/")[1]
        temp_name = img_name.split('/')[3]
        temp = temp_name.split('.')
        # temp_name=temp_name.split('.')[-2]
        if temp[-1] == 'jpg':
            temp[-1] = 'bmp'
        temp_name = '.'.join(temp)
        new_name.append(img_name.split('/')[0])
        new_name.append(new)
        new_name.append(img_name.split('/')[2])
        new_name.append(temp_name)
        img_name = '/'.join(new_name)
    # print(folder_name)

    # print(img_name)
    # path = os.getcwd() + '/NIR-VIS-2.0/'

    path = root_path  # sgong
    # print(os.path.join(path, img_name))

    #img = cv2.imread(os.path.join(path, img_name), cv2.IMREAD_GRAYSCALE)
    img=Image.open(os.path.join(path, img_name))
    print(os.path.join(path, img_name))

 
    if img is None:
        print(os.path.join(path, img_name))
        pp = os.path.join(path, img_name).split('\\')
        temp = pp[-1].split('.')
        temp = '.'.join(temp)
        print(temp)
        pp[-1] = temp
        i_p = '/'.join(pp)
        print('ip:'+i_p)
        img = Image.open(i_p)#cv2.imread(i_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print('image not found')
            print(i_p)
            exit()

    img=img.convert(("L"))
    img = transform(img)
    input[0, :, :, :] = img

    input = input.cuda()
    with torch.no_grad():
        input_var = torch.autograd.Variable(input)
    features= model(input_var)[1]

    return features.data.cpu().numpy()[0], int(folder_name)
#------------------------------------------------------------------------------------


def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list


def save_feature(save_path, img_name, features):
    # print(save_path, img_name)
    i_n = img_name.split('\\')
    # print(i_n)
    # i_n[1] = i_n[1] +'_feat'
    if 'NIR' in i_n[1]:
        # print(i_n[1])
        i_n[1] = 'NIR_crop6_feat'
    else:
        i_n[1] = 'VIS_crop6_feat'
    # folder_name = '/'.join(i_n[0:-1])
    # print(folder_name)
    # exit()

    i_n[-1] = i_n[-1].split('.')[0]
    img_name = '/'.join(i_n)

    # print(img_name)

    img_path = os.path.join(save_path, img_name)
    img_dir = os.path.dirname(img_path) + '/';
    # print(img_dir)

    # exit()
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # fname = os.path.splitext(img_path)[0]
    fname = img_path + '.npy'

    np.save(fname, features)


if __name__ == '__main__':
    main()
