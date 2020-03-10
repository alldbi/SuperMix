# SuperMix: Supervising the Mixing Data Augmentation

![](https://github.com/alldbi/KDA/blob/master/examples/fig1.png)

<p></p>

**Augment a dataset using the supervision of a teacher**

## Running SuperMix on ImageNet

```
python3 supermix.py --dataset imagenet --model resnet34 --save_dir ./outputdir --bs 16 --aug_size 50000 --w 16 --sigma 2
```
## Running SuperMix on CIFAR-100

1. Download the pretrained model by: 

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which saves the models to `save/models`
   
2. Run supermix.py




## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands in `scripts/run_cifar_distill.sh`. An example of running Geoffrey's original Knowledge Distillation (KD) is given by:

    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
    Therefore, the command for running CRD is something like:
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1
    ```
    
    Combining a distillation objective with KD is simply done by setting `-a` as a non-zero value, which results in the following example (combining CRD with KD)
    ```
    python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1     
    ```

3. (optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`

Note: the default setting is for a single-GPU training. If you would like to play this repo with multiple GPUs, you might need to tune the learning rate, which empirically needs to be scaled up linearly with the batch size, see [this paper](https://arxiv.org/abs/1706.02677)

## Benchmark Results on CIFAR-100:

Performance is measured by classification accuracy (%)

1. Performance of distillation vs. size of the augmented dataset when teacher and student are from the same architecture.

| Teacher/Student | Aug  | 0   | 50K | 100k |200k |300k | 400k | 500k |
| -----------     | ---- | ----| --- | --- | ---- | ---- | ---- | ---- |
| wrn-40-2/wrn-16-2    | MAS  | 73.25  |74.61±0.15  | 75.81±0.14 | 75.91±0.21 | 76.21±0.14 | 76.30±0.15 | 76.30±0.14 |                     
| resnet110/resnet20   | MAS  | 69.06  |  71.21±0.13          | 71.52±0.43 | 71.79±0.22 | 71.81±0.29 | 72.31±0.20 | 72.39±0.06 |
| vgg13/vgg8    | MAS  |  70.36    |  71.74±0.20  | 73.18±0.10 |   74.47±0.43   |   74.57±0.06   |   74.68±0.24   | 74.59±0.12  |


2. Performance of distillation vs. size of the augmented dataset when teacher and student are from different architecture.

| Teacher/Student      | Aug  |  0       | 50K | 100k |200k |300k | 400k | 500k |
| -----------          | ---- | -------  | --- | --- | ---- | ---- | ---- | ---- |
| vgg13/MobileNetV2    | MAS  |  64.60   | 68.13±0.18  | 69.42±0.39 | 69.26±0.80 | 70.68±0.48 | 69.76±0.65 | 70.49±0.56  |

3. Teacher and student are of the **same** architectural type.
    - MAA: KDA with unsupervised augmentation using averaging
    - MAS: KDA with supervised augmentation using the same Teacher network.  

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
| ImgNet| |  |   |    |  |  |  |  | 
| MixUp|  75.70±0.11     |   75.12±0.31     |   _71.78±0.26_    | _72.26±0.42_ |   73.70±0.32    |   76.17±0.12     |    74.07±0.32   |
| SuperMix|    **76.30±0.14**   |    **75.49±0.38**   |   _**72.13±0.46**_    | _**72.39±0.06**_ |   **74.25±0.07**    |   **76.92±0.08**    |   **74.59±0.12**    |

4. Teacher and student are of **different** architectural type.

| Teacher <br> Student | vgg13 <br> MobileNetV2 | ResNet50 <br> MobileNetV2 | ResNet50 <br> vgg8 | resnet32x4 <br> ShuffleNetV1 | resnet32x4 <br> ShuffleNetV2 | wrn-40-2 <br> ShuffleNetV1 |
|:---------------:|:-----------------:|:--------------------:|:-------------:|:-----------------------:|:-----------------------:|:---------------------:|
| Teacher <br> Student |    74.64 <br> 64.60    |      79.34 <br> 64.60     |  79.34 <br> 70.36  |       79.42 <br> 70.50       |       79.42 <br> 71.82       |      75.61 <br> 70.50      |
| KD | 67.37 | 67.35 | 73.81 | 74.07 | 74.45 | 74.83 |
| FitNet | 64.14 | 63.16 | 70.69 | 73.59 | 73.54 | 73.73 |
| AT | 59.40 | 58.58 | 71.84 | 71.73 | 72.73 | 73.32 |
| SP | 66.30 | 68.08 | 73.34 | 73.48 | 74.56 | 74.52 |
| CC | 64.86 | 65.43 | 70.25 | 71.14 | 71.29 | 71.38 |
| VID | 65.56 | 67.57 | 70.30 | 73.38 | 73.40 | 73.61 |
| RKD | 64.52 | 64.43 | 71.50 | 72.28 | 73.21 | 72.21 |
| PKT | 67.13 | 66.52 | 73.01 | 74.10 | 74.69 | 73.89 |
| AB | 66.06 | 67.20 | 70.65 | 73.55 | 74.31 | 73.34 |
| FT | 61.78 | 60.99 | 70.29 | 71.75 | 72.50 | 72.03 |
| NST | 58.16 | 64.96 | 71.28 | 74.12 | 74.68 | 74.89 |
| CRD | 69.73 | 69.11 | 74.30 | 75.11 | 75.65 | 76.05 |
| MixUp |  **70.53±0.21**  |   70.83±0.61    |    74.94±0.42  |   77.18±0.19   |   77.99±0.15   |   75.90±0.09     |
| SuperMix |  70.49±0.56   |    **71.69±0.36**   |   **75.45±0.08**   |    **77.69±0.32**  |   **78.66±0.20**   |    **76.88±0.35**    |
## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{tian2019crd,
  title={Contrastive Representation Distillation},
  author={Tian, Yonglong and Krishnan, Dilip and Isola, Phillip},
  journal={arXiv preprint arXiv:1910.10699},
  year={2019}
}
```
For any questions, please contact Yonglong Tian (yonglong@mit.edu).

## Acknowledgement

Thanks to Baoyun Peng for providing the code of CC and to Frederick Tung for verifying our reimplementation of SP. Thanks also go to authors of other papers who make their code publicly available.
