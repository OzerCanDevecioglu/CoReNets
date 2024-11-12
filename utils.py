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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torchaudio
from scipy.fft import fft, fftfreq, fftshift
from torch_stoi import NegSTOILoss
import math
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import math


    
class ImageTransform:
    def __init__(self, img_size=256):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
                lambda x: 2*(x-0.5)
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                lambda x: 2*(x-0.5)

                # transforms.Normalize(mean=[0.5], std=[0.5])
            ])}

    def __call__(self, img, phase='train'):
        img = self.transform[phase](img)

        return img





# Monet Dataset ---------------------------------------------------------------------------
class UWDataset(Dataset):
    def __init__(self, base_img_paths, style_img_paths,  transform, phase='train'):
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):        
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]
        # print(base_img_path)
        # print(style_img_path)
        base_img = Image.open(base_img_path)
        style_img = Image.open(style_img_path)

        base_img = self.transform(base_img, self.phase)
        style_img = self.transform(style_img, self.phase)

        return base_img, style_img
    
    
    
    
    
    
# Data Module
class UWDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, phase='train', seed=0):
        super(UWDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.phase = phase
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = glob.glob(os.path.join(self.data_dir, 'Corrupted', '*.png'))
        self.style_img_paths = glob.glob(os.path.join(self.data_dir, 'Clean', '*.png'))

    def train_dataloader(self):
        # random.seed()
        # random.shuffle(self.base_img_paths)
        # random.shuffle(self.style_img_paths)
        # random.seed(self.seed)
        self.train_dataset = UWDataset(self.base_img_paths, self.style_img_paths, self.transform, self.phase)
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory=True
                         )    
    

    
    

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    
    
    


def reshaper(a):
    b=a.values.reshape(len(a)*1000,1)
    b=b[:len(b)]
    c=b.reshape(int(len(b)/1000),1000)
    return c


    
def plot_loss_fig(a, strname,interval):
    plt.figure()
    plt.plot(range(interval, interval + len(a)*interval, interval), a)  # starting from 10, incrementing by 10
    plt.grid()
    # plt.legend()
    plt.show()
    plt.savefig(strname)
    plt.close()
    
def write_loss(aa,e,strname):
    with open(strname, 'a') as f:
                f.write("\n"+"Epoch : "+str(e)+ "  TrainLoss :"+str(aa))

def snr_loss(prediction, target):

    signal_power = torch.sum(target ** 2)

    noise = target - prediction
    noise_power = torch.sum(noise ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)

    snr_loss = -1* snr

    return snr_loss             
                
def PSNR_loss(a,b):
    
    psnr_list=[]
    for ii in range(len(a)):
   
        psg=np.abs(a[ii,:]) /  np.max(a[ii,:])
        psr=np.abs(b[ii,:]) /  np.max(b[ii,:])
        mse = np.mean((psg - psr) ** 2)
        res=10 * math.log10(1. / mse)
        psnr_list.append(res)
    return psnr_list




def psnr_loss(prediction, target, max_val=1.0):
    # Calculate the MSE (Mean Squared Error) between prediction and target
    
    prediction = prediction * 0.5 + 0.5
    target = target * 0.5 + 0.5
    
    mse = torch.mean((prediction - target) ** 2)

    # Calculate the PSNR (Peak Signal-to-Noise Ratio) using the formula:
    # PSNR = 20 * log10(max_val) - 10 * log10(mse)
    psnr = 10 * torch.log10((max_val**2) / mse)

    return -1 * psnr


def SNR_loss(b, a):
    snr_list = []

    for ii in range(len(a)):
        signal = a[ii, :] 
        noise = a[ii, :] - b[ii, :]  
        # signal_power = np.sum(signal ** 2)
        # noise_power = np.sum(noise ** 2) + 0.000000001
        snr = 20 * np.log10(np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(noise**2)))
        # snr = 10 * np.log10(signal_power / noise_power)
        snr_list.append(snr)

    return snr_list

def criterion_mse(y_true, y_pred):

    aaa = (y_true - y_pred)**2

    mse = np.mean(aaa)

    return mse


def calculate_PSNR(original_img, reconstructed_img):
    max_pixel_value = 255  # Maximum pixel value for an 8-bit image
    original_img = original_img * 0.5 + 0.5
    original_img = original_img * 255.0
    original_img = original_img.astype(int)

    
    reconstructed_img = reconstructed_img * 0.5 + 0.5
    reconstructed_img = reconstructed_img * 255.0
    reconstructed_img = reconstructed_img.astype(int)

    
    # print(np.max(original_img))
    # print(np.max(reconstructed_img))
    # Ensure images are numpy arrays
    original_img = np.array(original_img)/np.max(original_img)
    reconstructed_img = np.array(reconstructed_img)/np.max(reconstructed_img)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_img - reconstructed_img) ** 2)

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')  # PSNR is infinite if MSE is zero
    else:
        psnr = 10 * np.log10((1) / mse)

    return psnr

def calculate_SNR(signal, noise=None, typ='db', noisy=False):
    if noise is None and noisy:
        raise ValueError("If 'noisy' is True, 'noise' must be provided.")

    if noisy:
        signal = signal - noise

    if typ == 'db':
        ms=signal-signal.mean()
        mn=noise-noise.mean()
        SNR = 10 * np.log10(np.sum(ms**2) / np.sum(mn**2))
        
        # SNR = 20 * np.log10(np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(noise**2)))
    elif typ == 'amp':
        SNR = np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(noise**2))
    else:
        raise ValueError("Invalid value for 'typ'. Use 'db' or 'amp'.")

    return SNR


def denorm_image(temp):
    
    temp = temp.permute(1, 2, 0).detach().numpy()
    temp = temp * 0.5 + 0.5
    temp = temp * 255.0
    temp = temp.astype(int)

    
    return temp




def plot_images(App,Ma,e,d,x):

    
    data_dir2 = x
    transform = ImageTransform(img_size=256)

    dm2 = UWDataModule(data_dir2, transform, batch_size=10, phase='train')

    dm2.prepare_data()
    dataloader2 = dm2.train_dataloader()
   
    App.eval()
    Ma.eval()
  
    psnr_list=[]
    psnr_list2=[]
    psnr_list3=[]
    with torch.no_grad():
        for base, style in dataloader2:
            output = App(base.cuda()).cpu()
            outputpsnr = Ma(output.cuda()).cpu()
            inputpsnr = Ma(base.cuda()).cpu()
    
            # print(outputpsnr.size())
             
        a=0
        for verstappen in range(3):
            fig = plt.figure()
            fig.set_size_inches(15, 15)
            a=a+1
            plt.subplot(331)
            plt.imshow(denorm_image(base[a,:,:,:]))
            plt.title("Input Image PSNR: " + str(round(calculate_PSNR(style[a,:,:,:].detach().cpu().numpy(), base[a,:,:,:].detach().cpu().numpy()),2))+"|  Predicted PSNR: "+ str(np.round(inputpsnr[a].detach().cpu().numpy()*40,2)))
            plt.subplot(332)
            plt.imshow(denorm_image(style[a,:,:,:]))
            plt.title("GT Image")
            plt.subplot(333)
            plt.imshow(denorm_image(output[a,:,:,:]))
            plt.title("Output Image PSNR: " + str(round(calculate_PSNR(style[a,:,:,:].detach().cpu().numpy(), output[a,:,:,:].detach().cpu().numpy()),2))+"|  Predicted PSNR: "+ str(np.round(outputpsnr[a].detach().cpu().numpy()*40,2)))
            a=a+1
            plt.subplot(334)
            plt.imshow(denorm_image(base[a,:,:,:]))
            plt.title("Input Image PSNR: " + str(round(calculate_PSNR(style[a,:,:,:].detach().cpu().numpy(), base[a,:,:,:].detach().cpu().numpy()),2))+"|  Predicted PSNR: "+ str(np.round(inputpsnr[a].detach().cpu().numpy()*40,2)))
            plt.subplot(335)
            plt.imshow(denorm_image(style[a,:,:,:]))
            plt.title("GT Image")
            plt.subplot(336)
            plt.imshow(denorm_image(output[a,:,:,:]))
            plt.title("Output Image PSNR: "+ str(round(calculate_PSNR(style[a,:,:,:].detach().cpu().numpy(), output[a,:,:,:].detach().cpu().numpy()),2))+"|  Predicted PSNR: "+ str(np.round(outputpsnr[a].detach().cpu().numpy()*40,2)))
            
            a=a+1
            plt.subplot(337)
            plt.imshow(denorm_image(base[a,:,:,:]))
            plt.title("Input Image PSNR: " + str(round(calculate_PSNR(style[a,:,:,:].detach().cpu().numpy(), base[a,:,:,:].detach().cpu().numpy()),2))+"|  Predicted PSNR: "+ str(np.round(inputpsnr[a].detach().cpu().numpy()*40,2)))
            plt.subplot(338)
            plt.imshow(denorm_image(style[a,:,:,:]))
            plt.title("GT Image")
            plt.subplot(339)
            plt.imshow(denorm_image(output[a,:,:,:]))
            plt.title("Output Image PSNR: "+ str(round(calculate_PSNR(style[a,:,:,:].detach().cpu().numpy(), output[a,:,:,:].detach().cpu().numpy()),2))+"|  Predicted PSNR: "+ str(np.round(outputpsnr[a].detach().cpu().numpy()*40,2)))
            
            
            
            plt.show()
            plt.savefig(d+'Epoch_'+ str(e)+ '_PSNR_'+str(a)+'.png')
            plt.close()
            
    
    
    
    
    
    
