import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from fastonn import SelfONN2d as SelfONN2dlayer
from utils import UWDataset, UWDataModule,init_weights
from Model_details import Upsample,Downsample,Apprentice_Unet_Regressor,Master_Regressor
import seaborn as sn
from scipy.stats import norm
import scipy.signal as sig
import copy
import scipy.io as sio
from torch.autograd import Variable
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchaudio
from scipy.fft import fft, fftfreq, fftshift
from utils import reshaper, plot_loss_fig, write_loss, PSNR_loss,calculate_SNR,snr_loss, ImageTransform, calculate_PSNR,denorm_image,plot_images,psnr_loss
from focal_frequency_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class



app_loss_train=[]
mas_loss_train=[]
tr_mas_loss_gt=[]
tr_mas_loss_fk=[]


snr_sig_train=[]
snr_spec_train=[]


app_loss_val=[]
mas_loss_val=[]
val_mas_loss_gt=[]
val_mas_loss_fk=[]

snr_sig_val=[]
snr_spec_val=[]


data_dir = "train_set"

transform = ImageTransform(img_size=256)

dm2 = UWDataModule(data_dir,transform, batch_size=1, phase='test')
dm2.prepare_data()
dataloader = dm2.train_dataloader()

App = Apprentice_Unet_Regressor().cuda()
Ma = Master_Regressor().cuda()

    
num_epoch = 5000
lr=0.00001
betas=(0.5, 0.999)


Eps=3
Beta=0.05
Phi=100


App_params = list(App.parameters())
Ma_params = list(Ma.parameters())

optimizer_g = torch.optim.Adam(App_params, lr=lr, betas=betas)
optimizer_d = torch.optim.Adam(Ma_params, lr=lr*2, betas=betas)
criterion_psnr = psnr_loss

criterion_l1 = nn.L1Loss()
interval=10
targetpsnr=40 
E=0.00001    
  
for e in range(1,num_epoch):
    print("Epoch: "+str(e))
    App.train()
    Ma.train()
    
    predicted = []
    predicted=pd.DataFrame(data=predicted)
    inpdata = []
    inpdata=pd.DataFrame(data=inpdata)
    GT = []
    GT=pd.DataFrame(data=GT)
    
    tr_apprentice_loss=[]
    tr_master_loss=[]
    tr_master_loss_fk=[]
    tr_master_loss_gt=[]

    signal_snr=[]
    spec_snr=[]
    psnr_list=[]
    
    for input_img, real_img in (dataloader): 
      if(0):
       
          plt.subplot(211)
          plt.imshow(denorm_image(input_img[0,:,:,:].cpu().detach()))
          plt.title("Corrupted/Clean")
          plt.subplot(212)
          plt.imshow(denorm_image(real_img[0,:,:,:].cpu().detach()))
          
        
              
      input_img=input_img.cuda()

      real_img=real_img.cuda()
      real_label = torch.ones(input_img.size()[0], 1, 1).cuda()
      
      
      
      # Apprentice Loss
      fake_img = App(input_img).cuda()
      fake_img_ = fake_img.detach() 
      out_fake = Ma(fake_img).cuda()
      alpha=calculate_PSNR(real_img.detach().cpu().numpy(),fake_img.detach().cpu().numpy())
      alpha_hat=alpha/targetpsnr
      
      loss_g_bce = criterion_l1(out_fake, real_label)
      loss_g_mae = criterion_psnr(fake_img, real_img) 
      loss_g_dim = ffl(fake_img, real_img)  # calculate focal frequency loss
      loss_g = Eps*loss_g_bce + Beta * loss_g_mae +Phi * loss_g_dim
      
      
      optimizer_g.zero_grad()
      optimizer_d.zero_grad()
      loss_g.backward()
      optimizer_g.step()
      
      
      # Master Loss
      out_real = Ma(real_img)
      loss_d_real = criterion_l1(out_real, real_label)
      out_fake = Ma(fake_img_)
      loss_d_fake = criterion_l1(out_fake, real_label*alpha_hat)
      
      loss_d = loss_d_real + loss_d_fake 
      master_gt_l=targetpsnr*loss_d_real
      master_fk_l=targetpsnr*loss_d_fake
      master_tt_l=targetpsnr*loss_d

      

      optimizer_g.zero_grad()
      optimizer_d.zero_grad()
      loss_d.backward()
      optimizer_d.step()
      tr_apprentice_loss.append(loss_g.item())
      
      
      
      tr_master_loss.append(master_tt_l.item())
      tr_master_loss_gt.append(master_gt_l.item())
      tr_master_loss_fk.append(master_fk_l.item())
      
      
      signal_snr.append(loss_g_mae.item())
      spec_snr.append(loss_g_dim.item())
   
      
    write_loss(loss_g_bce,e,'loss_g_bce.txt')
    write_loss(loss_g_mae,e,'loss_g_mae.txt')
    write_loss(loss_g_dim,e,'loss_g_dim.txt')
      
   
    tr_apprentice_loss_mean=np.mean(tr_apprentice_loss)
    tr_master_loss_mean=np.mean(tr_master_loss)
    tr_mas_loss_gt_mean=np.mean(tr_master_loss_gt)
    tr_mas_loss_fk_mean=np.mean(tr_master_loss_fk)

      
    signal_snr_mean=np.mean(signal_snr)
    spec_snr_mean=np.mean(spec_snr)
    
    psnr_list=tr_apprentice_loss_mean
    app_loss_train.append(tr_apprentice_loss_mean)
    mas_loss_train.append(tr_master_loss_mean)
    tr_mas_loss_fk.append(tr_mas_loss_gt_mean)
    tr_mas_loss_gt.append(tr_mas_loss_fk_mean)

    
    
    snr_sig_train.append(signal_snr_mean)
    snr_spec_train.append(spec_snr_mean)


    write_loss(psnr_list,e,'TrainLoss.txt')
    if e%interval == 0:
        plot_images(App,Ma,e,"tr_figs/",'train_set')    
    plot_loss_fig(app_loss_train,"plots/Train_Apprentice.png",1)
    plot_loss_fig(tr_mas_loss_fk,"plots/Train_Master_fk.png",1)
    plot_loss_fig(mas_loss_train,"plots/Train_Master_total.png",1)
    plot_loss_fig(tr_mas_loss_gt,"plots/Train_Master_GT.png",1)
    
    plot_loss_fig(tr_master_loss_fk,"plotsres/Train_Master_FK_res"+str(e)+".png",1)
    plot_loss_fig(tr_master_loss_gt,"plotsres/Train_Master_GT_res"+str(e)+".png",1)
    
    plot_loss_fig(snr_sig_train,"plots/SNR_sig_train.png",1)
    plot_loss_fig(snr_spec_train,"plots/SNR_spec_train.png",1)
                   
    if e%interval == 0:       
          data_dir = "val_set"
          dm2 = UWDataModule(data_dir,transform, batch_size=1, phase='test')
          dm2.prepare_data()
          dataloader3 = dm2.train_dataloader()
          
         
          # &
          predicted = []
          predicted=pd.DataFrame(data=predicted)
          inpdata = []
          inpdata=pd.DataFrame(data=inpdata)
          GT = []
          GT=pd.DataFrame(data=GT)
          
          val_apprentice_loss=[]
          val_master_loss=[]
          val_master_loss_fk=[]
          val_master_loss_gt=[]
          
          App.eval()
          Ma.eval()
          
          vsignal_snr=[]
          vspec_snr=[]
          vpsnr_list=[]
          with torch.no_grad():
              for input_img, real_img in (dataloader3): 
              
                input_img=input_img.cuda()
                real_img=real_img.cuda()
                real_label = torch.ones(input_img.size()[0], 1, 1).cuda()
                # Apprentice Loss
                fake_img = App(input_img).cuda()
                fake_img_ = fake_img.detach() 
                out_fake = Ma(fake_img).cuda()
                
                alpha=calculate_PSNR(real_img.detach().cpu().numpy(),fake_img.detach().cpu().numpy())
                alpha_hat=alpha/targetpsnr
                loss_g_bce = criterion_l1(out_fake, real_label)
                loss_g_mae = criterion_psnr(fake_img, real_img) 
                
                loss_g_dim = ffl(fake_img, real_img)  # calculate focal frequency loss

                loss_g = Eps*loss_g_bce + Beta * loss_g_mae +Phi*loss_g_dim
                
                # Master Loss
                out_real = Ma(real_img)
                loss_d_real = criterion_l1(out_real, real_label)
                out_fake = Ma(fake_img_)
                loss_d_fake = criterion_l1(out_fake, real_label*alpha_hat)
                loss_d = loss_d_real + loss_d_fake 
                
                master_gt_l=targetpsnr*loss_d_real
                master_fk_l=targetpsnr*loss_d_fake
                master_tt_l=targetpsnr*loss_d

                val_apprentice_loss.append(loss_g.item())
                val_master_loss.append(master_tt_l.item())
                
                val_master_loss_gt.append(master_gt_l.item())
                val_master_loss_fk.append(master_fk_l.item())
                
                vsignal_snr.append(loss_g_mae.item())
                vspec_snr.append(loss_g_dim.item())
    
          
              val_apprentice_loss_mean=np.mean(val_apprentice_loss)
              val_master_loss_mean=np.mean(val_master_loss)
              
              val_mas_loss_gt_mean=np.mean(val_master_loss_gt)
              val_mas_loss_fk_mean=np.mean(val_master_loss_fk)
             
              vsignal_snr_mean=np.mean(vsignal_snr)
              vspec_snr_mean=np.mean(vspec_snr)
              
              psnr_list=val_apprentice_loss_mean
              app_loss_val.append(val_apprentice_loss_mean)
              mas_loss_val.append(val_master_loss_mean)
              
              val_mas_loss_fk.append(val_mas_loss_gt_mean)
              val_mas_loss_gt.append(val_mas_loss_fk_mean)
              
              
              
              
              snr_sig_val.append(vsignal_snr_mean)
              snr_spec_val.append(vspec_snr_mean)
    
    
              write_loss(psnr_list,e,'ValLoss.txt')
              if e%interval == 0:
                  plot_images(App,Ma,e,"val_figs/",'val_set')    
              
              plot_loss_fig(app_loss_val,"plots/Val_Apprentice.png",interval)
              
              plot_loss_fig(val_mas_loss_fk,"plots/Val_Master_fk.png",interval)
              plot_loss_fig(mas_loss_val,"plots/Val_Master_total.png",interval)
              plot_loss_fig(val_mas_loss_gt,"plots/Val_Master_GT.png",interval)
              
              plot_loss_fig(tr_master_loss_fk,"plotsres/Train_Master_FK_res"+str(e)+".png",interval)
              plot_loss_fig(tr_master_loss_gt,"plotsres/Train_Master_GT_res"+str(e)+".png",interval)
              
              plot_loss_fig(snr_sig_val,"plots/SNR_sig_val.png",interval)
              plot_loss_fig(snr_spec_val,"plots/SNR_spec_val.png",interval)
                             
              torch.save(App.state_dict(), 'weights/model_weights_'+str(e)+'_.pth')
              torch.save(Ma.state_dict(), 'weightsmaster/model_weights_'+str(e)+'_.pth')
    
       