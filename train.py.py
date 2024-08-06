import torch
import os
import sys
import time
# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
import argparse
from utils.utils_SH import *
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from DVGFace.dataset import Dataset
from DVGFace.my1027trydataset import tryDataset
from DVGFace.visdataset import visDataset
import cv2
from torch.autograd import Variable
import sys
from model.defineHourglass_1024_gray_skip_matchFeature import *
from DVGutils import *
from DVGlightcnn import LightCNN_29v2
CUDA_VISBLE_DEVICES=0,1
class Model(nn.Module):
    def __init__(self):
        super().__init__()   # 继承父类的所有属性
        self.linear_1 = nn.Linear(1, 256)  # 输入20列的特征（数据集决定），输出为64（假设给64个隐藏层）1,512
        self.linear_2 = nn.Linear(256, 128)  # 接上层64输入，输出64维512,256
        self.linear_4 = nn.Linear(128, 256)
        self.linear_3 = nn.Linear(256, 1)  # 因为是逻辑回归，只要一个输出
        self.relu = nn.ReLU()
        self.c=nn.Conv2d(9,9,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        #print(input.size())

        x1 = self.linear_1(input)
        #print(x1)
        x2 = self.relu(x1)
        #print(x2)
        x3 = self.linear_2(x2)
        x4 = self.relu(x3)
        x5 = self.linear_4(x4)
        x6 = self.relu(x5)
        x7 = self.linear_3(x6)
        #x8= self.relu(x7)#yesterday ok:delete thie line
        #print(x5.size())
        #x8 = self.sigmoid(x7)
        #x6=self.c(x5)
        #print(x6.size())
        #print(x5)
        return x7

class loss_vc(nn.Module):
    def __init__(self,lvc_knel):
        super(loss_vc, self).__init__()
        #self.lvc_knel=lvc_knel
        #lvc_knel=
        kernel = torch.ones(1,1,lvc_knel,lvc_knel)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x,x0,chunknum,x3,x4):
        #batch_size = x.shape[0]


        # transform_matrix = np.array([[0.257, 0.564, 0.098],
        #                              [-0.148, -0.291, 0.439],
        #                              [0.439, -0.368, -0.071]])
        # shift_matrix = np.array([16, 128, 128])
        #
        # ycbcr_image = np.zeros(shape=x.shape)
        # w, h, _ = x.shape
        # # 2：遍历每个像素点的三个通道进行变换
        # for i in range(w):
        #     for j in range(h):
        #         ycbcr_image[i, j, :] = np.dot(transform_matrix, x[i, j, :]) + shift_matrix
        # #print(ycbcr_image.shape)
        # ycbcr_image=torch.from_numpy(x)#ycbcr_image)
        # #print(ycbcr_image)
        # x1 = ycbcr_image[:, :,0]
        # x2 = ycbcr_image[:, :,1]
        # x3 = ycbcr_image[:, :,2]
        # print('x1',x1)
        # print('x2',x2)
        #print(x1)
        # x1 = torch.clamp(x1, min=0, max=1)
        # x2 = torch.clamp(x2, min=0, max=1)
        # x3 = torch.clamp(x3, min=0, max=1)
        # oriImg = x0[0].cpu().data.numpy()
        # oriImg = oriImg.transpose((1, 2, 0))
        # oriImg = np.squeeze(oriImg)
        # oriImg = (oriImg * 255.0).astype(np.float32)
        #x=torch.from_numpy(x).cpu()
        #x0=torch.from_numpy(oriImg)
        # print(x)
        # print(x0)
        # x1=Variable(torch.unsqueeze(x, dim=0).float())
        # x2 = Variable(torch.unsqueeze(x0, dim=0).float())
        # print(x1)
        # print(x2)
        # x3 = Variable(torch.unsqueeze(x3, dim=0).float())
        #-------------------------------------------------------------
        # result1=torch.chunk(input=x,chunks=chunknum, dim=3)
        # result2=torch.chunk(input=x0,chunks=chunknum, dim=3)
        # chunks=chunknum
        # sum=0
        # for i in range(chunks):
        #
        #
        #     # print(len(result))
        #     # print(result[1])
        #     # print(result[1].size())
        #     # x1 = torch.clamp(result1[i], min=0, max=1)
        #     # x2 =torch.clamp(result2[i], min=0, max=1)
        #     # x3 = torch.clamp(x3, min=0, max=1)
        #     #print(x1.size())
        #     #print(x2)
        #
        #
        #     x1 = F.conv2d(result1[i], self.weight, stride=1).reshape(-1)
        #     x2 = F.conv2d(result2[i], self.weight, stride=1).reshape(-1)
        #     # print(x1)
        #     # print(x2)
        #     # x3 = F.conv2d(x3.unsqueeze(1), self.weight, stride=1).reshape(-1)
        #     x1 = torch.var(x1, unbiased=True)
        #     x2 = torch.var(x2, unbiased=True)
        #     # x3 = torch.var(x3, unbiased=True)
        #     x = x1-x2#loss_vc1(x1,x2) #+ x2 + x3
        #     # x = F.conv2d(x,self.weight,stride=14)
        #     x=torch.abs(x)
        #     x = x.reshape(-1)
        #     x = torch.mean(x)
        #     #print(x)
        #     sum+=x
        #     #print(sum)
        # sum=sum/chunks
        #-------------------------------------------------

        x1 = F.conv2d(x, self.weight, stride=1).reshape(-1)
        x2 = F.conv2d(x0, self.weight, stride=1).reshape(-1)
        # print(x1)
        # print(x2)
        # x3 = F.conv2d(x3.unsqueeze(1), self.weight, stride=1).reshape(-1)
        x1 = torch.var(x1, unbiased=True)
        x2 = torch.var(x2, unbiased=True)
        # x3 = torch.var(x3, unbiased=True)
        x = x2 - x1  # loss_vc1(x1,x2) #+ x2 + x3
        # x = F.conv2d(x,self.weight,stride=14)
        x = torch.abs(x)
        x = x.reshape(-1)
        sumn = torch.mean(x)

        x3 = F.conv2d(x3, self.weight, stride=1).reshape(-1)
        x4 = F.conv2d(x4, self.weight, stride=1).reshape(-1)
        # print(x1)
        # print(x2)
        # x3 = F.conv2d(x3.unsqueeze(1), self.weight, stride=1).reshape(-1)
        x3 = torch.var(x3, unbiased=True)
        x4 = torch.var(x4, unbiased=True)
        # x3 = torch.var(x3, unbiased=True)
        xv = x4 - x3  # loss_vc1(x1,x2) #+ x2 + x3
        # x = F.conv2d(x,self.weight,stride=14)
        xv = torch.abs(xv)
        xv = xv.reshape(-1)
        sumv = torch.mean(xv)
        #--------------------------------------
        sum=sumn+sumv
        return sum

def loss_fn(chunks,lvc_knel,beta,lamda,lrvc,iteration256,iteration,light_pred,loss_fnn1,loss_fnn2,loss_fn3,loss_fnn4,loss_fnn5,loss_fnn6,visiteration,vislight_pred,my_network,loss_vc1):
    # DVG_feature--------------------------------------

    from sklearn.metrics.pairwise import cosine_similarity
    DVGmodel = LightCNN_29v2(num_classes=725)
    load_model(DVGmodel, "/data/DVG2/DVG-Face-master/pre_train/LightCNN_29Layers_V2_checkpoint.pth")#/data/DPR-master/LightCNN128_epoch_15.pth.tar
    DVGmodel.eval()
    # -----------------------------------------
    iteration_clone=iteration.clone()
    visiteration_clone=visiteration.clone()
    light_pred_clone=light_pred.clone()
    vislight_pred_clone=vislight_pred.clone()
    outputImg, _, _, _ = my_network(iteration_clone, light_pred_clone, 0,iteration_clone)
    visoutputImg, _, _, _ = my_network(visiteration_clone, vislight_pred_clone, 0, visiteration_clone)
    # print(iteration_clone.shape)
    #print(outputImg)
    # outputImgg=outputImg.detach()

    oo_feature = DVGmodel(iteration.clone())[1]
    pp_feature = DVGmodel(outputImg.clone())[1]
    vis_feature=DVGmodel(visiteration.clone())[1]
    vispp_feature = DVGmodel(visoutputImg.clone())[1]
    # print(oo_feature)
    # print(pp_feature)
    # o_feature = oo_feature.data.cpu().numpy()[0]
    # p_feature = pp_feature.data.cpu().numpy()[0]
    #
    #
    #
    # o_image=o_feature
    # p_image=p_feature
    # image_oo=np.array(o_image).reshape(1, -1)
    # image_pp=np.array(p_image).reshape(1, -1)
    # loss0=cosine_similarity(image_oo,image_pp).T
    # ofeature_tensor = torch.from_numpy(image_oo)
    # pfeature_tensor = torch.from_numpy(image_pp)
    #loss1=torch.from_numpy((loss0)).cuda()
    loss_flag=torch.ones([1]).cuda()
    #print(loss_flag)
    loss1=loss_fnn1(oo_feature,pp_feature,loss_flag)
    #print(loss1)weight

    # print(iteration)
    # print(outputImg)

    # loss=0.2*loss1+8*loss2
    # if epoch < 25:
    #     loss = loss1
    # else:
    #     loss = loss2
    # print(loss)
    loss_flag2 = torch.ones([1]).cuda()
    loss2=loss_fn3(pp_feature,vis_feature,loss_flag2)
    loss3 = loss_fnn2(iteration, outputImg)  # .cuda()

    loss_flag3 = torch.ones([1]).cuda()
    loss4 = loss_fnn4(vis_feature, vispp_feature, loss_flag3)
    loss6 = loss_fnn5(visiteration, visoutputImg)
    loss_flag4 = torch.ones([1]).cuda()
    loss5 = loss_fnn6(vispp_feature, oo_feature, loss_flag4)
    # print(outputImg.shape)
    # print(iteration_clone.shape)
    # outputImg = outputImg[0].cpu().data.numpy()
    # outputImg = outputImg.transpose((1, 2, 0))
    # outputImg = np.squeeze(outputImg)
    #outputImg = (outputImg * 255.0).astype(np.float32)
    # resultLab = cv2.cvtColor(outputImg, cv2.COLOR_GRAY2BGR)
    # resultLab = cv2.cvtColor(resultLab, cv2.COLOR_BGR2YCR_CB)
    # print(outputImg)
    # iteration_clone = iteration_clone[0].cpu().data.numpy()
    # iteration_clone = iteration_clone.transpose((1, 2, 0))
    # iteration_clone = np.squeeze(iteration_clone)
    # iteration_clone = (iteration_clone * 255.0).astype(np.float32)
    lossvc = loss_vc(lvc_knel)
    #print(outputImg)
    # print( iteration_clone)

    lossv = lossvc(outputImg, iteration256,chunks,visiteration_clone,visoutputImg)#,loss_vc1
    loss=loss1+loss2*beta+loss3*lamda+loss4+loss5*beta+loss6*lamda+lrvc*lossv
    return loss1,loss2*beta,loss3*lamda,loss,loss4,loss5,loss6,lossv#,loss



#loss_fn = nn.BCELoss()
    #num_of_batches = len(data)//batch
def main():
    # nn.Module:继承这个类
    # __init__:初始化所有层weight
    # forward:定义模型的运算过程
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', default='/data/MTCNN_Portable/CASIA_O_MTCNN',
                        type=str)  # /data/MTCNN_Portable/CASIA_O_MTCNN#/data/HFGI-main/CASIA_DLIB
    parser.add_argument('--train_list', default='/data/DPR-master/CASIA_relight/relight_probe_256time.txt',
                        type=str)  # /data/DPR-master/DVGFace/trainlist.txt#/data/DPR-master/CASIA_relight/relight_probe_256time.txt#/data/HFGI-main/CASIA_DLIB/myprotocols.txt
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr1', default=1, type=float)
    parser.add_argument('--lr2', default=1, type=int)
    # load model
    modelFolder = 'trained_model/'

    my_network_512 = HourglassNet(16)
    my_network = HourglassNet_1024(my_network_512, 16)
    my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_1024_03.t7')))
    my_network.cuda()
    my_network.eval()
    for p in my_network.parameters():
        p.requires_grad = False

    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)
    # 每次运行得到模型和其优化方法
    lr = 0.0001
    model = Model().cuda()
    model=torch.nn.DataParallel(model,device_ids=[0,1]).cuda()
    #model.train()
    #opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=1e-4)  #
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # #进行模型参数初始化
    # def init_weights1(model):
    #     if isinstance(model, nn.Linear):
    #         nn.init.xavier_uniform_(model.weight)
    #         model.bias.data.fill_(0.1)#ok:0.1
    # model.apply(init_weights1)

    global args
    args = parser.parse_args()
    # model, optim = get_model()
    # train_loader = torch.utils.data.DataLoader(Dataset(args),batch_size=args.batch_size, shuffle=None, pin_memory=None)



    # 定义loss
    loss_fnn1 = nn.CosineEmbeddingLoss()
    loss_fnn2 = nn.MSELoss()  ##MSE
    loss_fn3 = nn.CosineEmbeddingLoss()  # nn.L1Loss()
    loss_fnn4 = nn.CosineEmbeddingLoss()
    loss_fnn5 = nn.MSELoss()  ##MSE
    loss_fnn6 = nn.CosineEmbeddingLoss()  # nn.L1Loss()
    loss_vc1=nn.L1Loss()
    epochs = 50
    lightFolder = 'data/example_light/'
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)
    mag = np.sqrt(x**2 + z**2)
    valid = mag <=116
    y = -np.sqrt(np.abs(1 - (x*valid)**2 - (z*valid)**2))
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_02.txt'))
    sh = sh[0:9]
    sh = sh * 0.95#0.95

    # rendering half-sphere
    sh = np.squeeze(sh)
    # print(sh)
    shading = get_shading(normal, sh)
    # print(shading)
    value = np.percentile(shading, 98)  # 95
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    # ----------------------------------------------
    #  rendering images using the network
    # ----------------------------------------------
    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).cuda())
    # print(iteration)
    # 开始训练
    num=0
    loss=0.0
    loss_list=[]
    loss1_list=[]
    loss2_list=[]
    loss3_list=[]
    loss4_list = []
    loss5_list = []
    loss6_list = []
    lossvc_list = []
    beta=20#args.lr2#20#
    lamda=args.lr2#100#
    lrvc=25#args.lr2#25
    lvc_knel=4#4#args.lr2
    chunks=1#8#args.lr2
    for epoch in range(epochs):
        model.train()

        for niritem,visiteration in zip(tryDataset(args),visDataset(args)):#enumerate(train_loader, start=1):
            #loss=0.0
            num=num+1
            #with torch.no_grad():
            iteration=niritem[1]
            iteration256=niritem[0]
            _, _, outputSH, _ = my_network(iteration.clone(), sh, 0,iteration.clone())
            #print(outputSH.shape)
            _, _, visoutputSH, _ = my_network(visiteration.clone(), sh, 0, visiteration.clone())
            #print(outputSH)
            #outSH=outputSH
            light_pred = model(outputSH)
            light_pred.require_grad=True
            vislight_pred = model(visoutputSH)
            vislight_pred.require_grad = True
            #iteration.require_grad=True
            loss1,loss2,loss3,loss,loss4,loss5,loss6,lossv= loss_fn(chunks,lvc_knel,beta,lamda,lrvc,iteration256,iteration,light_pred,loss_fnn1,loss_fnn2,loss_fn3,loss_fnn4,loss_fnn5,loss_fnn6,visiteration,vislight_pred,my_network,loss_vc1)
            # loss1.require_grad = True
            # loss2.require_grad = True
#            , loss4, loss5, loss6
            # print(np.shape(iteration))
            # print(np.shape(outputImg))
            # print(np.shape(o_feature))
            # print(np.shape(p_feature))

            #print(loss)
            #loss.require_grad = True
            #loss=loss/32
            loss.backward()

            if num%8==0:
                opt.step()
                opt.zero_grad()
        # 打印出每个epoch的loss情况
        loss_list.append(loss)
        loss1_list.append(loss1)
        loss2_list.append(loss2)
        loss3_list.append(loss3)
        loss4_list.append(loss4)
        loss5_list.append(loss5)
        loss6_list.append(loss6)
        lossvc_list.append(lossv)
        with torch.no_grad():
            print('epoch:', epoch,'loss:', loss.data.item(),'loss1:',loss1.data.item(),'loss2:',loss2.data.item(),'loss3:',loss3.data.item(),'loss4:',loss4.data.item(),'loss5:',loss5.data.item(),'loss6:',loss6.data.item(),'lossvc:',lossv.data.item())
        if epoch%49==0:
            savename='LightCNNepoch'+str(epoch+1)+'202418dlibAdvvisnir'+str(lamda)+str(beta)+'lossvc'+str(lvc_knel)+'knelvis'+str(lrvc)+'yesvischunk'+str(chunks)+'.pt'
            rootpath='/data/DPR-master/Adv_checkpoint'
            savename=os.path.join(rootpath,savename)
            print(savename)
            torch.save(model.state_dict(),savename)

    x = np.arange(0, 50, step=1)
    # y=np.arange(0,50,step=1)
    # plt.figure(figsize=(5,5))
    plt.subplot(331)
    plt.plot(x, loss_list, label='loss')
    plt.subplot(332)
    plt.plot(range(epochs), lossvc_list, label='lossvc')
    plt.subplot(334)
    plt.plot(range(epochs), loss1_list, label='loss1')
    plt.subplot(335)
    plt.plot(range(epochs), loss2_list, label='loss2')
    plt.subplot(336)
    plt.plot(range(epochs), loss3_list, label='loss3')
    plt.subplot(337)
    plt.plot(range(epochs), loss4_list, label='loss4')
    plt.subplot(338)
    plt.plot(range(epochs), loss5_list, label='loss5')
    plt.subplot(339)
    plt.plot(range(epochs), loss6_list, label='loss6')

    #sname = ''
    sname = '/data/DPR-master/pltloss/' + str(lamda) + str(beta) + 'loss+vc'+str(lvc_knel)+'knelvis'+ str(lrvc) + 'yesvisdata202418chunk'+str(chunks)+'LightCNN.png'
    plt.savefig(sname)
    plt.close()
    print('pltsavename:' + sname)
    # plt.yticks(np.arange(0,10,step=1))
    # plt.xticks(np.arange(0,50,step=1))
    # plt.show()

if __name__ == "__main__":

    main()