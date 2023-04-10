close all;clear all;
inp_path = 'G:\lmx\RDAFNet-main\log\deblur\results\real_world\';
inp_list = [dir(strcat(inp_path,'*.jpg')); dir(strcat(inp_path,'*.png'))];

img_num = length(inp_list);

niqe_total = 0;
nrisa_total = 0;
if img_num > 0 
    for i = 1:img_num
       inp_name = inp_list(i).name;
       input = imread(strcat(inp_path,inp_name));
       input_gray = rgb2gray(input);
       niqe_val = niqe(input);
       nrisa_val = final(input_gray);
       niqe_total = niqe_total + niqe_val;
       nrisa_total = nrisa_total + nrisa_val;
       fprintf('%d/%d, %s NIQE: %f NRISA: %f\n', i, img_num, inp_name, niqe_val, nrisa_val);
   end
end

niqe_mean = niqe_total / img_num;
nrisa_mean = nrisa_total / img_num;

fprintf('average NIQE:%f NRISA:%f\n', niqe_mean, nrisa_mean);

