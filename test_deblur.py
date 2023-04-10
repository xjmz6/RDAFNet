from utils.data_RGB import get_validation_data
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from utils.logger import *
import yaml
with open('options/test/deblur/SSRDAFNet.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

model_restoration = utils.get_arch(opt['MODEL'])

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['MODE'], opt['MODEL']['NAME'])
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
ckpt_dir   = os.path.join(model_dir, opt['VAL']['PRETRAIN_MODEL'])

utils.load_checkpoint(model_restoration, ckpt_dir)

print("===>Testing using weights of ", opt['VAL']['PRETRAIN_MODEL'])
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

psnr_val_rgb = []
for ii, data_val in enumerate(val_loader, 1):
    target_img = img_as_ubyte(utils.renorm(data_val[0]).numpy().squeeze().transpose((1,2,0)))
    input_ = data_val[1].cuda()


    with torch.no_grad():
        restored = model_restoration(input_)[-1]

    restored_img = img_as_ubyte(torch.clamp(utils.renorm(restored),0,1).cpu().numpy().squeeze().transpose((1,2,0)))

    psnr_val_rgb.append(PSNR(restored_img, target_img))
    print('%-6s \t %f' % (data_val[2][0], psnr_val_rgb[-1]))
    if opt['VAL']['SAVE_IMG']:
        save_img_path = os.path.join(result_dir, opt['VAL']['PRETRAIN_MODEL'][:-4])
        file_name = data_val[2][0]
        utils.mkdir(save_img_path)
        utils.save_img(os.path.join(save_img_path, file_name), restored_img)

avg_psnr  = sum(psnr_val_rgb)/ii
print('total images = %d \t avg_psnr = %f' % (ii,avg_psnr))
