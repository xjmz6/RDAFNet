## Training
- Download the [Datasets]()

- Train the model with default arguments by running

```
python train.py
```

## Evaluation

### Download the [model]() and place it in ./pretrained_models/

#### Testing on GoPro dataset
- Download [images]() of GoPro and place them in `./Datasets/GoPro/test/`
- Run
```python
python test.py --dataset GoPro
```

#### Testing on HIDE dataset
- Download [images]() of HIDE and place them in `./Datasets/HIDE/test/`
- Run
```python
python test.py
```


#### Testing on real-world blurred images dataset
- Download [images]() of RealBlur-J and place them in `./Datasets/RealBlur_J/test/`
- Run
```python
python deblur/test.py
```



#### To reproduce PSNR/SSIM scores of the paper on GoPro、HIDE and SPAD datasets, run this MATLAB script
```python
evaluate_PSNR_SSIM.m 
```

#### To reproduce NIQE/NRISA scores of the paper on real-world blurred images dataset, run
```python
evaluate_NIQE_NRISA.m
```
