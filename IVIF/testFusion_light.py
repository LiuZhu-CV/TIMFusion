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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from model.model_searched import  Network_Fusion9_Meta
from model.model_searched import Network_Fusion9_3
import genotypes

tag = 'Road'
data_name = 'fusion'  # DIDMDN_Data or DDN_Data
# -----------------------
data_root = 'E:/liuzhu/darts-master/data/vis_if'

tag = 'Road'
data_name = 'fusion'  # DIDMDN_Data or DDN_Data
# -----------------------
data_root = 'E:/liuzhu//data/vis_if'
data_test = '../data/' + data_name + '/test_syn/'
imlist_pth = '../lists/' + data_name + '/trainlist.txt'
test_pth = '../lists/' + data_name + '/Roadtest.txt'

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
def load(model, model_path):
  model.load_state_dict(torch.load(model_path,map_location='cuda:0'),strict=False)
if __name__ == '__main__':

    genotype = eval("genotypes.%s" % 'fusion_light')
    genotype_1 = eval("genotypes.%s" % 'fusion_ir_proposed')
    genotype_2 = eval("genotypes.%s" % 'fusion_vis_proposed')

    cells_name = ['Cell_Chain2', 'Cell_Chain2']
    cells_name2 = ['Cell_Chain2']
    model_1 = Network_Fusion9_Meta(48, 1, genotype, cells_name)
    # model_2 = Network_Fusion9_3(48, 1, cells_name2, [genotype_1, genotype_2])
    # model_1.load_state_dict(torch.load('$$$。pt'),strict= False)
    model_1 = model_1.cuda()
    # model_2 = model_2.cuda()
    load(model_1, '../checkpoints/IVIF/light/weights_1_1169.pt')
    # utils.load(model_2, './single_3DC/weights_2_1099.pt')
    # utils.load(model_1, './fusion0128/weights_1_1049.pt')
    # utils.load(model_2, './fusion0128/weights_2_1049.pt')
    model_1 = model_1.eval()
    # model_2 = model_2.eval()
    file_ = open('../TNO_test.txt', 'r')
    lines = file_.readlines()
    for line in lines:

        pth1, pth2, new_h, new_w = line.split('>>')
        print(pth1,pth2,new_h,new_w)
        img_name = pth1.split('/')[-1].split('.')[0]
        print(img_name)
        im_input = cv2.imread( '.'+pth1)[:, :, 0]
        new_h = int(new_h)
        new_w = int(new_w)
        im_input = np.array(im_input)[np.newaxis, np.newaxis, :] / 255.
        im_input = torch.tensor(im_input).type(torch.FloatTensor)
        lr = Variable(im_input, requires_grad=False).cuda()

        im_input2 = cv2.imread('.'+ pth2)[:, :, 0]

        im_input2 = np.array(im_input2)[np.newaxis, np.newaxis, :] / 255.
        im_input2 = torch.tensor(im_input2).type(torch.FloatTensor)

        vis = Variable(im_input2, requires_grad=False).cuda()
        print(np.shape(vis),np.shape(lr))
        with torch.no_grad():
            fused_1 = model_1(vis,lr)
            # f_ir, f_vis, fused = model_2(lr,vis, fused_1)
            res = fused_1.data.cpu().numpy()
            _,c, h, w= np.shape(res)
            torchvision.utils.save_image(fused_1[:,:,0:h-new_h,0:w-new_w],'./road_'+img_name+'.png')
        # res *= 255
        # res = res[0].transpose((1, 2, 0))
        # h,w,c = np.shape(res)
        # print(h,w,c)
        # res = res[0:h-new_h,0:w-new_w][:,:,0]
        # print(np.shape(res))
        # res = np.array(res).astype(np.uint8)
        # if not os.path.exists('./Darts2_vis-ir'):
        #     os.mkdir('./Darts2_vis-ir')
        # Image.fromarray(res).save('./alb_op/our2/road_'+img_name+'.png')


    print('Test done.')
