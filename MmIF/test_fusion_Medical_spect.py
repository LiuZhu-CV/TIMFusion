import os, sys
sys.path.insert(1, '../')

from model.model_searched import Network_Fusion9_Meta, Network_Fusion9_2
import torch
import cv2
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image


import genotypes

def load(model, model_path):
  model.load_state_dict(torch.load(model_path,map_location='cuda:0'),strict=False)

tag = 'Road'
data_name = 'fusion'
# -----------------------
data_root = 'E:/liuzhu/darts-master/data/vis_if'
data_test = '../data/' + data_name + '/test_syn/'
imlist_pth = '../lists/' + data_name + '/trainlist.txt'
test_pth = '../lists/' + data_name + '/Roadtest.txt'

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()

if __name__ == '__main__':
    genotype = eval("genotypes.%s" % 'fusion_M')
    genotype_1 = eval("genotypes.%s" % 'fusion_mri_spect')
    genotype_2 = eval("genotypes.%s" % 'fusion_spect')
    cells_name = ['Cell_Fusion', 'Cell_Chain2']
    cells_name2 = ['Cell_Chain2']
    model_1 = Network_Fusion9_Meta(48, 1, genotype, cells_name)
    model_2 = Network_Fusion9_2(48, 1, cells_name2, [genotype_1, genotype_2])

    model_1 = model_1.cuda()
    model_2 = model_2.cuda()

    load(model_1, '../checkpoints/MRISPECT/weights_1_1149.pt')
    load(model_2, '../checkpoints/MRISPECT/weights_2_1149.pt')

    #读取
    # file_ = open('../lists/fusion/mrcttest4.txt','r')
    # lines = file_.readlines()
    for index in range(243, 267):
        # pth1, pth2 = line.split('>>')
        # # folder = pth1.split('/')[2]
        # img_name = pth1.split('/')[-1].split('.')[0].split('_')[1]

        im_input = cv2.imread('../2-png/' + str(index) + '.png')
        im_ = cv2.cvtColor(im_input, cv2.COLOR_BGR2YUV)
        im_input = im_[:, :, 0]

        print(np.shape(im_input))
        im_input = np.array(im_input)[np.newaxis, np.newaxis, :] / 255.
        # print(np.shape(im_input), im_input)
        im_input = torch.tensor(im_input).type(torch.FloatTensor)
        im_input = Variable(im_input, requires_grad=False).cuda()
        ###############################################
        print('../1-png/' + str(index) + '.png')
        im_input2 = plt.imread('../1-png/' + str(index) + '.png', 0)
        print(np.shape(im_input2))
        im_input2 = np.array(im_input2[:, :, 0])[np.newaxis, np.newaxis, :] / 255.
        # print(np.shape(im_input), im_input)
        im_input2 = torch.tensor(im_input2).type(torch.FloatTensor)
        im_input2 = Variable(im_input2, requires_grad=False).cuda()

        with torch.no_grad():
            fused_1 = model_1(im_input2, im_input)
            f_ir, f_vis, fused = model_2(im_input2, im_input, fused_1)
        res = fused.data.cpu().numpy()
        print('res',np.shape(res))
        res[res > 1] = 1
        res[res < 0] = 0
        res *= 255
        res = res.astype(np.uint8)[0]
        res = res.transpose((1, 2, 0))
        #crop_image
        h,w,c = np.shape(res)
        print(h,w,c)
        print(np.shape(res))
        im_[:,:,0] =res[:,:,0]
        if not os.path.exists('./oursa3_spect1112_/'):
            os.mkdir('./oursa3_spect1112_/')

        cv2.imwrite('./oursa3_spect1112_/' + str(index) + '.png',
                    cv2.cvtColor(im_, cv2.COLOR_YUV2BGR))


    print('Test done.')
