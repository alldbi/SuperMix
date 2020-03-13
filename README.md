# SuperMix: Supervising the Mixing Data Augmentation

![](https://github.com/alldbi/KDA/blob/master/examples/fig1.png)

<p></p>

**Pytorch implementation of SuperMix [paper](https://arxiv.org/abs/2003.05034), which is a supervised method for data augmentation.**

## Run SuperMix

- Auguments are:
    * `--dataset`: specify the dataset, choices: `imagenet` or `cifar100`, default: `cifar100`.
    * `--model`: specify the supervisor for augmentation. For `cifar100`, all the models in 'models/\_\_init\_\_.py' can be used. For imagenet, all the models in `torchvision.models` can be used.
    * `--device`: specify the device, default: `cuda:0`.
    * `--save_dir`: the directory to save the mixed images.  
    * `--input_dir`: the input directory of the imagenet dataset.
    * `--bs`: batch size, default: `100`. 
    * `--aug_size`: number of mixed images to produce, default: `500000`.
    * `--k`: number of input images to be mixed, default: `2`.
    * `--max_iter`: maximum number of iterations on each batch, default: `50`.
    * `--alpha`: alpha value for the Dirichlet distribution, default: `3`.
    * `--sigma`: standard deviation of the Guassian smoothing function, default: `1`.
    * `--w`: spatial size of the mixing masks, default: `8`.
    * `--lambda_s`: multiplier for the sparsity loss, default: `25`.
    * `--tol`: percentage of successfull samples in the batch for early termination, default: `70`.
    * `--plot`: plot the mixed images after generation, default: `True`


### Run on the ImageNet data
1. Run supermix.py
```
python3 supermix.py --dataset imagenet --model resnet34 --save_dir ./outputdir --bs 16 --aug_size 50000 --w 16 --sigma 2
```
2. Sample outputs

<p align="center"> 
<img src="https://github.com/alldbi/KDA/blob/master/examples/imagenet.png">
</p>


### Run on the CIFAR-100 data

1. Download the pretrained model by: 

```
sh scripts/fetch_pretrained_teachers.sh
```
   which saves the models to `save/models`
   
2. Run supermix.py

```
python3 supermix.py --dataset cifar100 --model resnet110 --save_dir ./outputdir --bs 64 --aug_size 50000 --w 8 --sigma 1
```

3. Sample outputs

<p align="center"> 
<img src="https://github.com/alldbi/KDA/blob/master/examples/cifar100.png">
</p>

## Evaluating SuperMix for knowledge distillation and object classification

**Code for the distillation is forked/copied from [the official code of CRD](https://github.com/HobbitLong/RepDistiller)**

1. Fetch the pretrained teacher models by:

   ```
   sh scripts/fetch_pretrained_teachers.sh
   ```
   which will download and save the models to `save/models`
   
2. Produce augmented data using SuperMix by: 

   ```
   python3 supermix.py --dataset cifar100 --model resnet110 --save_dir ./output --bs 128 --aug_size 500000 --w 8 --sigma 1
   ```   

3. Run the distillation
- using cross-entropy (Equation 9 in the paper) by:

   ```
   python3 train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet20 --distill kd --model_s resnet8x4 -r 2.0 -a 0 -b 0 --aug_type supermix --aug_dir ./output --trial 1
   ```
- using the original distillation objective proposed by Hinton et. al., (Equation 8 in the paper) by:

   ```
   python3 train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet20 --distill kd --model_s resnet8x4 -r 1.8 -a 0.2 -b 0 --aug_type supermix --aug_dir ./output --trial 1
   ```


- where the flags are explained as:
   - `--path_t`: specify the path of the teacher model
   - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
   - `--distill`: specify the distillation method
   - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
   - `-a`: the weight of the KD loss, default: `None`
   - `-b`: the weight of other distillation losses, default: `None`
   - `--aug_type`: type of the augmentation, choices: `None`, `supermix`, `mixup`, `cutmix`.
   - `--aug_dir`: the directory of augmented images when `supermix` is selected for `aug_type`.
   - `--aug_alpha`: alpha for the Dirichlet distribution when `mixup` or `cutmix` is selected for `aug_type`. 
   - `--trial`: specify the experimental id to differentiate between multiple runs.
   

4. (optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`

Note: the default setting is for a single-GPU training. If you would like to play this repo with multiple GPUs, you might need to tune the learning rate, which empirically needs to be scaled up linearly with the batch size, see [this paper](https://arxiv.org/abs/1706.02677)

## Benchmark Results on CIFAR-100:

Performance is measured by classification accuracy (%) 

| Teacher <br> Student | wrn-40-2 <br> wrn-16-2 | wrn-40-2 <br> wrn-40-1 | resnet56 <br> resnet20 | resnet110 <br> resnet20 | resnet110 <br> resnet32 | resnet32x4 <br> resnet8x4 |  vgg13 <br> vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:------------------:|:------------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    75.61 <br> 73.26    |    75.61 <br> 71.98    |    72.34 <br> 69.06    |     74.31 <br> 69.06    |     74.31 <br> 71.14    |      79.42 <br> 72.50     | 74.64 <br> 70.36 |
| KD | 74.92 | 73.54 | 70.66 | 70.67 | 73.08 | 73.33 | 72.98 |
| FitNet | 73.58 | 72.24 | 69.21 | 68.99 | 71.06 | 73.50 | 71.02 |
| AT | 74.08 | 72.77 | 70.55 | 70.22 | 72.31 | 73.44 | 71.43 |
| SP | 73.83 | 72.43 | 69.67 | 70.04 | 72.69 | 72.94 | 72.68 |
| CC | 73.56 | 72.21 | 69.63 | 69.48 | 71.48 | 72.97 | 70.71 |
| VID  | 74.11 | 73.30 | 70.38 | 70.16 | 72.61 | 73.09 | 71.23 |
| RKD  | 73.35 | 72.22 | 69.61 | 69.25 | 71.82 | 71.90 | 71.48 |
| PKT  | 74.54 | 73.45 | 70.34 | 70.25 | 72.61 | 73.64 | 72.88 |
| AB   | 72.50 | 72.38 | 69.47 | 69.53 | 70.98 | 73.17 | 70.94 |
| FT   | 73.25 | 71.59 | 69.84 | 70.22 | 72.37 | 72.86 | 70.58 |
| FSP  | 72.91 | 0.00 | 69.95 | 70.11 | 71.89 | 72.62 | 70.23 |
| NST  | 73.68 | 72.24 | 69.60 | 69.53 | 71.96 | 73.30 | 71.53 |
| CRD  | 75.48 | 74.14 | 71.16 | 71.46 | 73.48 | 75.51 | 73.94 |
| CRD+KD |  75.64| 74.38| 71.63 | 71.56 | 73.75 | 75.46 | 74.29 |
| ImgNet32| 74.91 | 74.80 | 71.38 | 71.48 | 73.17 | 75.57 | 73.95 |
| MixUp|  76.20| 75.53 | 72.00 | 72.27 | 74.60 | 76.73 | 74.56 |
| CutMix| 76.40 | 75.85 | 72.33 | 72.68 | 74.24 |76.81 | 74.87 |
| SuperMix|**76.93**|**76.11**|**72.64**|**72.75** |   **74.80**    |   **77.16**    |   **75.38**    |
| ImgNet32+KD| 76.52 | 75.70 | 72.22 | 72.23 | 74.24 | 76.46 | 75.02 |
| MixUp+KD| 76.58 | 76.10 | 72.89 | 72.82 | 74.94 | 77.07 | 75.58 |
| CutMix+KD| 76.81 | 76.45 | 72.67 | 72.83 | 74.87 | 76.90 | 75.50 |
| SuperMix+KD| **77.45**   |**76.53**| **73.19**| **72.96** | **75.21**|   **77.59**    |   **76.03**    |


## Citation

If you found SuperMix helpful for your research, please cite our paper: 

```
@article{dabouei2020,
  title={SuperMix: Supervising the Mixing Data Augmentation},
  author={Dabouei, Ali and Soleymani, Sobhan and Taherkhani, Fariborz and Nasrabadi, Nasser M},
  journal={arXiv preprint arXiv:2003.05034},
  year={2020}
}
```
Moreover, if you are developing distillation methods, we encourage you to cite CRD, due to their notable contribution by benchmarking the state-of-the-art methods of distillation.
```
@inproceedings{tian2019crd,
  title={Contrastive Representation Distillation},
  author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

