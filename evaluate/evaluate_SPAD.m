clear all;
inp_path = 'G:\lmx\RDAFNet-main\log\derain\results\SPAD\';
tar_path =  'G:\SPAD_UNZIP\spa-data\test\small\norain\';

inp_list = ([dir(strcat(inp_path,'*.jpg')); dir(strcat(inp_path,'*.png'))]);
tar_list = ([dir(strcat(tar_path,'*.jpg')); dir(strcat(tar_path,'*.png'))]);
inp_files = sort_nat({inp_list.name});
tar_files = sort_nat({tar_list.name});

img_num  = length(inp_list);

psnr_total = 0;
ssim_total = 0;

for i=1:img_num                         % the number of testing samples
   inp_name = inp_files{i};
   tar_name = tar_files{i};
   x_true  = im2double(imread(strcat(tar_path, tar_name)));
   x_true = rgb2ycbcr(x_true);
   x_true = x_true(:,:,1); 
   x = im2double(imread(strcat(inp_path, inp_name)));
   x = rgb2ycbcr(x);
   x = x(:,:,1);
   psnr_val = psnr(x,x_true);
   ssim_val = ssim(x,x_true);
   psnr_total = psnr_total + psnr_val;
   ssim_total = ssim_total + ssim_val;
   fprintf('%d/%d image_name %s gt_name %s PSNR: %f SSIM: %f\n', i, img_num, inp_name, tar_name, psnr_val, ssim_val);
end
fprintf('psnr=%6.4f, ssim=%6.4f\n',psnr_total/img_num, ssim_total/img_num)