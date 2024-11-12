import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import UWDataset, UWDataModule,init_weights
from Model_details import Upsample,Downsample,Apprentice_Unet_Regressor,Master_Regressor
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils import reshaper, plot_loss_fig, write_loss, PSNR_loss,calculate_SNR,snr_loss, ImageTransform, calculate_PSNR,denorm_image,plot_images,psnr_loss
from focal_frequency_loss import FocalFrequencyLoss as FFL
from PIL import Image


transform = ImageTransform(img_size=256)


G_basestyle = Apprentice_Unet_Regressor()
checkpoint =torch.load("weights/model_weights_830_.pth")

G_basestyle.load_state_dict(checkpoint)

G_basestyle.eval()
G_basestyle=G_basestyle.cuda()


data_dir = "val_set/"
dm2 = UWDataModule(data_dir,transform, batch_size=1, phase='test')
dm2.prepare_data()
dataloader3 = dm2.train_dataloader()




# &
flag=1
vpsnr_list=[]
masterpsnr_list=[]
i=0
for input_img, real_img in (dataloader3): 
  i=i+1
  print(i)
  input_img=input_img.cuda()
  real_img=real_img.cuda()
  fake_img =G_basestyle(G_basestyle(G_basestyle(input_img))).cuda()
  fake_img_ = fake_img.detach()  
  alpha=calculate_PSNR(real_img.detach().cpu().numpy(),fake_img.detach().cpu().numpy())
  vpsnr_list.append(alpha)
  
  
  
  if(flag):
      fake_img1 = fake_img[0].cpu().detach()
      ddd=denorm_image(fake_img1)
      ddd = ddd.astype(np.uint8)
      output_image = Image.fromarray(ddd)
      output_image.save('results/'+str(i) +'.jpg')
     
# plt.plot() 
  
print(np.mean(vpsnr_list))

# total_params = sum(p.numel() for p in G_basestyle.parameters())
# print("Total number of parameters in G_basestyle model:", total_params)

# total_params_in_millions = total_params / 10**6
# print("Total number of parameters in G_basestyle model (in millions): {:.2f} M".format(total_params_in_millions))
