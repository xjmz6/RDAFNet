close all;clear all;
inp_path = 'G:\lmx\RDAFNet_three_stage\three_GOPRO_2950\';
tar_path =  'D:\lmx\cal_PSNR\GOPRO\test\sharp\';

inp_list = ([dir(strcat(inp_path,'*.jpg')); dir(strcat(inp_path,'*.png'))]);
tar_list = ([dir(strcat(tar_path,'*.jpg')); dir(strcat(tar_path,'*.png'))]);
inp_files = sort_nat({inp_list.name});
tar_files = sort_nat({tar_list.name});

img_num  = length(inp_list);

psnr_total = 0;
ssim_total = 0;
if img_num > 0 
    for i = 1:img_num
       inp_name = inp_files{i};
       tar_name = tar_files{i};
       input  = imread(strcat(inp_path, inp_name));
       target = imread(strcat(tar_path, tar_name));
       psnr_val = psnr(input, target);
       ssim_val = ssim(input, target);
       psnr_total = psnr_total + psnr_val;
       ssim_total = ssim_total + ssim_val;
       fprintf('%d/%d image_name %s gt_name %s PSNR: %f SSIM: %f\n', i, img_num, inp_name, tar_name, psnr_val, ssim_val);
   end
end

psnr_mean = psnr_total / img_num;
ssim_mean = ssim_total / img_num;

fprintf('average PSNR: %f SSIM: %f\n', psnr_mean, ssim_mean);

