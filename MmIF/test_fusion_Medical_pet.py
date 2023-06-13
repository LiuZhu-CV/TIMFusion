
import os, sys

sys.path.insert(1, '../')
import torch
import cv2
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image



from model.model_searched import Network_Fusion9_Meta, Network_Fusion9_2

# DDN_Data contains a few huge images. You need "module parallel" to run the code on it.
# ------- Option -------psnr: 27.928875194139582
# ssim: 0.8776472609357657
tag = 'Road'
data_name = 'fusion'  # DIDMDN_Data or DDN_Data
# -----------------------

data_root = 'E:/liuzhu/darts-master/data/vis_if'
data_test = '../data/' + data_name + '/test_syn/'
imlist_pth = '../lists/' + data_name + '/trainlist.txt'
test_pth = '../lists/' + data_name + '/Roadtest.txt'


def load(model, model_path):
  model.load_state_dict(torch.load(model_path,map_location='cuda:0'),strict=False)
# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()

if __name__ == '__main__':
    genotype = eval("genotypes.%s" % 'fusion_M')
    genotype_1 = eval("genotypes.%s" % 'fusion_mri_pet')
    genotype_2 = eval("genotypes.%s" % 'fusion_pet')
    cells_name = ['Cell_Fusion', 'Cell_Chain2']
    cells_name2 = ['Cell_Chain2']
    model_1 = Network_Fusion9_Meta(48, 1, genotype, cells_name).cuda()
    model_2 = Network_Fusion9_2(48, 1, cells_name2, [genotype_1, genotype_2]).cuda()

    model_1 = model_1.eval()
    model_2 = model_2.eval()
    load(model_1, './weights_1_1149.pt')
    load(model_2, './weights_2_1149.pt')

    #读取
    # file_ = open('../lists/fusion/mrcttest4.txt','r')
    # lines = file_.readlines()
    for index in range(163,179):
        # pth1, pth2 = line.split('>>')
        # # folder = pth1.split('/')[2]
        # img_name = pth1.split('/')[-1].split('.')[0].split('_')[1]

        im_input = cv2.imread('../medical/MR-PET_1/PET/'+str(index)+'.png')
        im_ = cv2.cvtColor(im_input,cv2.COLOR_BGR2YUV)
        im_input = im_[:,:,0]

        print(np.shape(im_input))
        im_input = np.array(im_input)[np.newaxis, np.newaxis, :]/ 255.
        # print(np.shape(im_input), im_input)
        im_input = torch.tensor(im_input).type(torch.FloatTensor)
        im_input = Variable(im_input, requires_grad=False).cuda()
        ###############################################
        im_input2 = cv2.imread('../medical/MR-PET_1/MR/'+str(index)+'.png',0)
        print(np.shape(im_input2))
        im_input2 = np.array(im_input2)[np.newaxis, np.newaxis, :] / 255.
        im_input2 = torch.tensor(im_input2).type(torch.FloatTensor)
        im_input2 = Variable(im_input2, requires_grad=False).cuda()

        with torch.no_grad():
            fused_1 = model_1(im_input, im_input2)
            f_ir, f_vis, fused = model_2(im_input, im_input2, fused_1)
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
        if not os.path.exists('./our_pet_1110_2/'):
            os.mkdir('./our_pet_1110_2/')
        cv2.imwrite('./our_pet_1110_2/'+str(index)+'.png',cv2.cvtColor(im_,cv2.COLOR_YUV2BGR))


    print('Test done.')
