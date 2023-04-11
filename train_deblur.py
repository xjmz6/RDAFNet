import os
from utils.logger import *
import yaml
with open('.options/train/deblur/MSRDAFNet.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True
from visdom import Visdom
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import datetime
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR

import utils
from utils.data_RGB import get_training_data, get_validation_data
import utils.losses as losses


######### Logs dir ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['MODE'], opt['MODEL']['NAME'])
utils.mkdir(log_dir)

train_log = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt['MODEL'])
num_params = 0
with open(train_log,'a') as f:
    f.write(dict2str(opt)+'\n')
    f.write(str(model_restoration)+'\n')
    for param in model_restoration.parameters():
        num_params += param.numel()
    f.write('parameters:' + str(num_params))
    
from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(model_restoration, (3,256,256), as_strings=True,print_per_layer_stat = False, verbose=True)
log('macs = %s \t params = %s'%(macs, params),train_log)

torch.nn.parallel.DataParallel(model_restoration)
model_restoration.to(device)

######### Optimizer  ###########
start_epoch = 1
optimizer = torch.optim.Adam(model_restoration.parameters(), **opt['OPTIM'])
scheduler = LinearLR(optimizer, **opt['SCHE'])

if opt['TRAIN']['RESUME']:
        path_chk_rest = os.path.join(model_dir, opt['TRAIN']['PRETRAIN_MODEL'])
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt['OPTIM']['lr']
        for i in range(1, start_epoch-299):
            scheduler.step()
        new_lr = scheduler.get_last_lr()[0]
    
        log('------------------------------------------------------------------------------',train_log)
        log("==> Resuming Training with learning rate:", train_log)
        log('------------------------------------------------------------------------------',train_log)

######### Loss ###########
criterion_L1 = torch.nn.L1Loss()
criterion_vgg = losses.PerceptualLoss()

######### DataLoaders ###########
train_dataset = get_training_data(opt['PATH']['TRAIN_DATASET'], {'patch_size':opt['TRAIN']['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=4, drop_last=False, pin_memory=False)

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['TRAIN']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)

trainset_len = train_dataset.__len__()
valset_len = val_dataset.__len__()
total_iters = trainset_len//opt['TRAIN']['BATCH_SIZE'] if trainset_len%opt['TRAIN']['BATCH_SIZE'] == 0 else trainset_len//opt['TRAIN']['BATCH_SIZE']+1
log('trainset length: %d \ttotal iters per epoch: %d \tvalset length: %d'%(trainset_len, total_iters, valset_len), train_log)

######### Visdom ###########
print('==> visdom initial')
window_loss = Visdom(port=opt['PORT']) 
window_blur = Visdom(port=opt['PORT'])
window_restored = Visdom(port=opt['PORT'])
window_sharp = Visdom(port=opt['PORT'])
window_loss.line([0.], [1.], win='train_loss',opts=dict(title='train_loss',legend=['loss_L1'],xlabel='epoch',ylabel='loss'))

######### Train ###########
log("------------------------------------------------------------------",train_log)
log('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt['TRAIN']['TOTAL_EPOCHS']),train_log)
log("------------------------------------------------------------------",train_log)

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, opt['TRAIN']['TOTAL_EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for iter, data in enumerate(train_loader, 1):

        # zero_grad
        optimizer.zero_grad()
        
        target = data[0].cuda()
        input_ = data[1].cuda()
        
        r1, r2, restored = model_restoration(input_)

        # Compute loss 
        restor_fft = torch.fft.rfft2( restored, dim=(-2,-1))
        target_fft = torch.fft.rfft2( target, dim=(-2,-1))
        r1_fft = torch.fft.rfft2( r1, dim=(-2,-1))
        r2_fft = torch.fft.rfft2( r2, dim=(-2,-1))

        loss_fft = criterion_L1( restor_fft, target_fft) * 0.1 + criterion_L1( r1_fft, target_fft) * 0.1 + criterion_L1( r2_fft, target_fft) * 0.1
        loss_con = criterion_L1( restored, target) + criterion_L1( r1, target) + criterion_L1( r2, target)
        loss_vgg = criterion_vgg.get_loss(restored, target) * 0.01 + criterion_vgg.get_loss(r1, target) * 0.01 + criterion_vgg.get_loss(r2, target) * 0.01 

        loss = loss_con + loss_fft + loss_vgg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), 0.01)
        optimizer.step()
        epoch_loss +=loss.item()

        target = utils.renorm(target)
        input_ = utils.renorm(input_)
        restored = utils.renorm(restored)

        if iter % opt['TRAIN']['PRINT_FRE'] == 0:
            log("Epoch:%-4d \tIter:%-4d \tTime:%-4.f \tCON:%.6f \tFFT:%.6f \tVGG:%.6f \t\tLearningRate:%.10f"%(epoch, iter, time.time()-epoch_start_time, loss_con, loss_fft, loss_vgg, scheduler.get_last_lr()[0]),train_log)
            step = epoch + iter/total_iters
            window_loss.line([loss.item()], [step], win = 'train_loss', update='append')
            window_blur.images(input_[0], nrow=1, win='Blur_Train', opts=dict(title='Blur'))
            window_restored.images(torch.clamp(restored[0], 0, 1), nrow=1, win='Restored_Train', opts=dict(title='Restored'))
            window_sharp.images(target[0], nrow=1, win='Sharp_Train', opts=dict(title='Sharp'))
            
    log("------------------------------------------------------------------",train_log)
    log("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.10f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_last_lr()[0]),train_log)
    log("------------------------------------------------------------------",train_log)
    
    if epoch >= 300:
        scheduler.step()

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 
    
    #### Save ####
    if epoch % opt['TRAIN']['SAVE_FRE'] == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

    #### Evaluation ####
    if epoch % opt['TRAIN']['VAL_FRE'] == 0:
        
        ######### release training gpu memory ##########
        del input_
        del restored
        del target

        torch.cuda.empty_cache()

        log('eval epoch %d psnr'%(epoch), train_log)
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate(tqdm(val_loader), 1):
            target_img = img_as_ubyte(utils.renorm(data_val[0]).numpy().squeeze().transpose((1,2,0)))
            input_ = data_val[1].cuda()

            with torch.no_grad():
                r1, r2, restored = model_restoration(input_)[-1]

            restored_img = img_as_ubyte(torch.clamp(utils.renorm(restored),0,1).cpu().numpy().squeeze().transpose((1,2,0)))
            
            ######### release testing gpu memory ##########
            del restored
            del input_
            torch.cuda.empty_cache()

            psnr_val_rgb.append(PSNR(restored_img, target_img))
            log('%-6s \t %f' % (data_val[2][0], psnr_val_rgb[-1]), os.path.join(log_dir, 'val_' + str(epoch)+'.txt'), P=False)
            if opt['TRAIN']['SAVE_IMG']:
                save_img_path = os.path.join(result_dir, 'epoch_' + str(epoch))
                file_name = data_val[2][0] + '_restored.png'
                utils.mkdir(save_img_path)
                utils.save_img(os.path.join(save_img_path, file_name), restored_img)

        avg_psnr  = sum(psnr_val_rgb)/ii
        log('total images = %d \t avg_psnr = %f' % (ii,avg_psnr), os.path.join(log_dir, 'val_' + str(epoch)+'.txt'), P=False)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_epoch = epoch
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        log("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, avg_psnr, best_epoch, best_psnr),train_log)