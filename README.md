## Train

Demo of multi-stage RDAFNet on deblurring.

- Download the datasets modified your [dataset path](https://github.com/xjmz6/RDAFNet/blob/main/options/train/deblur/MSRDAFNet.yml#L26-L27)

- Train the multi-stage RDAFNet for deblurring with default arguments by running

```
python train_deblur.py
```

## Test

Demo on multi-stage RDAFNet.

### Download the pre-training models of [multi-stage RDAFNet](https://drive.google.com/drive/folders/1ixozsGTSX3JpJq2Bar_Ew8P9IuKwwFYl?usp=share_link) or [sing-stage RDAFNet](https://drive.google.com/drive/folders/1AqG7TwUyA81GmyvHD5O4IYXrcsEYdSVR?usp=share_link) .

#### Testing on GoPro and HIDE dataset
- Download the datasets and modify the [dataset path](https://github.com/xjmz6/RDAFNet/blob/main/options/test/deblur/MSRDAFNet.yml#L17) of the corresponding configuration file.

- Place the weight in `./log/deblur/MSRDAFNet/models/` and run

```python
python test_deblur.py 
```


#### Testing on real-world blurred images dataset
- Download the real-world blurred images dataset and modify the [dataset path](https://github.com/xjmz6/RDAFNet/blob/main/options/test/deblur/MSRDAFNet.yml#L17) of the corresponding configuration file.
- Place the weight in `./log/deblur/MSRDAFNet/models/` and run
```python
python test_deblur_NR.py 
```

#### Testing on SPAD dataset

- Download the SPAD dataset and modify the [dataset path](https://github.com/xjmz6/RDAFNet/blob/main/options/test/derain/MSRDAFNet.yml#L17) of the corresponding configuration file.
- Place the weight in `./log/derain/MSRDAFNet/models/` and run

```
python test_derain.py 
```

## evaluating

### Download RDAFNet's restoration results on [GoPro](https://drive.google.com/drive/folders/1gtz_SNEs5z_dHkM1qhyE6A1iNzCMs52d?usp=share_link), [HIDE](https://drive.google.com/drive/folders/1AUSS7xcJNIWPxqW4T46rP_Up6yg_Y9DM?usp=share_link), [real-world](https://drive.google.com/drive/folders/1vqKJG5p9vLUMj26O8jXczncsDCbe7pju?usp=share_link) and [SPAD](https://drive.google.com/file/d/1dlwZFKgNuaf2M4zVssQRoJFsGumI7nAc/view?usp=share_link) datasets.

#### To calculate PSNR/SSIM scores on GoPro、HIDE , run this MATLAB script

```matlab
evaluate_PSNR_SSIM.m 
```

#### To calculate NIQE/NRISA scores on real-world blurred images dataset, run this MATLAB script

```matlab
evaluate_NIQE_NRISA.m
```

#### To calculate NIQE/NRISA scores on SPAD, run this MATLAB script

```matlab
evaluate_SPAD.m 
```
