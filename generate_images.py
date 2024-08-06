'''
    this is a simple test file
'''
import sys
sys.path.append('model')
sys.path.append('utils')

from utils.utils_SH import *

# other modules
import os
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
from skimage import exposure
#------------------------------------------------------------------
import torchvision.transforms as transforms
from PIL import Image
from DVGutils import *
from DVGlightcnn import LightCNN_29v2
#from sklearn.metrics.pairwise import cosine_similarity

import os
# ---------------- create normal for rendering half sphere ------
#AP-ARA-------------------------------------------------------------------
from AP_ARA_model import Model
AP_Net=Model().cuda()
AP_Net=torch.nn.DataParallel(AP_Net,device_ids=[0,1]).cuda()#duoka
AP_Net.load_state_dict(torch.load(' '))
#/data/DPR-master/Adv_checkpoint/epoch50GCV.pt
#/data/DPR-master/Adv_checkpoint/epoch50CVC.pt

AP_Net.train(False)
#-------------------------------------
def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list

input = torch.zeros(1, 1, 256, 256)



def del_file(path):
    ls=os.listdir(path)
    for i in ls:
        c_path=os.path.join(path,i)
        os.remove(c_path)
#--------------------------------------------------------------

#-----------------------------------------------------------------

modelFolder = 'trained_model/'#''#'/data/DPR-master/Adv_checkpoint/'##

from model.defineHourglass_1024_gray_skip_matchFeature import *
my_network_512 = HourglassNet(16)
my_network = HourglassNet_1024(my_network_512, 16)
my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_1024_03.t7')))#epoch41mytrained_model_1024_03,100,20,0.t7##
my_network.cuda()
my_network.train(False)

lightFolder = 'data/new_light/'

saveFolder = '/data/DPR-master/CASIA_mtcnn_re/'
TempFolder='TEMP'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)
root_path = '/data/MTCNN_Portable/CASIA_O_MTCNN/'#'/data/HFGI-main/CASIA_DLIB/'#
#protocols = 'protocols/'
gallery_file_list ='/data/DPR-master/CASIA_relight/relight_probe_256time.txt'# '/data/HFGI-main/CASIA_DLIB/myprotocols.txt'#'nir_probe_1.txt'
gallery=read_list(gallery_file_list)#where 128protocols
#txtpath='/data/DPR-master/CASIA_relight/vis_train_time.txt'#'/data/DPR-master/CASIA_relight/relight_probe256time.txt'
#f=open(txtpath,'w')
zerocount=0
for imgname in gallery:
    print(imgname)
    features=[]
    new_name=[]

    new = imgname.split("/")[1]

    temp_name = imgname.split('/')[3]
    temp = temp_name.split('.')

    temp_name = '.'.join(temp)
    # print(temp_name)
    new_name.append(imgname.split('/')[0])
    new_name.append(new)
    new_name.append(imgname.split('/')[2])
    new_name.append(temp_name)
    imgname = '/'.join(new_name)
    #imgname = '/'.join(imgname.split('\\'))

    img = cv2.imread(root_path+imgname)

    #img = cv2.resize(img, (256, 256))
    row, col, _ = img.shape

    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())


    sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_01.txt'))
    sh = sh[0:9]
    sh = sh * 0.55#best:0.55

    #--------------------------------------------------
    # rendering half-sphere
    sh = np.squeeze(sh)

    #--------------------------------------------------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    #----------------------------------------------
    #  rendering images using the network
    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    nooutputImg, _, outputSH, _ = my_network(inputL, sh, 0,inputL)
    new_sh = AP_Net(outputSH)
    #print(new_sh)

    new_sh=new_sh*1.9
    outputImg, _, outputSH, _ = my_network(inputL, new_sh, 0,inputL)


    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg * 255.0).astype(np.uint8)
    outputImg = np.reshape(outputImg, (row, col))
    #print(outputImg)

    count=cv2.countNonZero(outputImg)
    print(count)
    if count<10000:
        #outputImg=exposure.adjust_gamma(outputImg,0.5)
        zerocount=zerocount+1

    lablename = imgname.split('/')[-2]
    numname=imgname.split('/')[-1]
    typename = imgname.split('/')[-3]
    sname = imgname.split('/')[-4]
    fold=os.path.join(saveFolder,sname,typename,lablename)
    print(fold)
    if not os.path.exists(fold):
        os.makedirs(fold)
    cv2.imwrite(os.path.join(saveFolder,imgname), outputImg)
   # f.write(os.path.join(sname,typename,lablename,numname)+'\n')
    # ------------------------------------------------------
    outputSH = new_sh[0].cpu().data.numpy()
    outputSH = outputSH.transpose((1, 2, 0))
    outputSH=outputSH[0][0]
    print(outputSH)
    o_shading = get_shading(normal, outputSH)
    #print(o_shading)
    value = np.percentile(o_shading, 98)  # 95
    ind = o_shading > value
    o_shading[ind] = value
    o_shading = (o_shading - np.min(o_shading)) / (np.max(o_shading) - np.min(o_shading))
    o_shading = (o_shading * 255.0).astype(np.uint8)
    o_shading = np.reshape(o_shading, (256, 256))
    o_shading = o_shading * valid
    print(o_shading)
    #cv2.imwrite(os.path.join(saveFolder,'light_o.png'),o_shading)
    lsame=imgname.split('.')[0]
    lsname=str(lsame)+'_light_o.png'
    cv2.imwrite(os.path.join(saveFolder,lsname), o_shading)
# -___--------------------------------------------------------------
    #del_file('/data/DPR-master/TEMP')
#f.close()
print('blackpictur:'+str(zerocount))