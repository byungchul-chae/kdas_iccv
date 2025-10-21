# [ICCV 2025] What to Distill? Fast Knowledge Distillation with Adaptive Sampling

## Abstract

Knowledge Distillation (KD) has been established as an effective technique for reducing the resource requirements of models when tackling computer vision tasks. Prior work has studied how to distill the knowledge of a teacher model better, but it overlooks how data affects the distillation result. This work examines the impact of data in knowledge distillation from two perspectives: (i) quantity of knowledge and (ii) quality of knowledge. Our examination finds that faster knowledge distillation can be achieved by using data with a large amount of high-quality knowledge in distillation. Based on the findings, this work proposes an efficient adaptive sampling method called KDAS for faster knowledge distillation, which enhances the distillation efficiency by selecting and applying ‘good’ samples for the distillation. This work shows that our adaptive sampling methods can effectively accelerate the training efficiency of a student model when combined with existing KD methods.

## KDAS (Knowledge Distillation with Adaptive Sampling)

KDAS is a method that enhances the efficiency of Knowledge Distillation through dynamic sampling and penalization.

The code is built on [mdistiller](<https://github.com/megvii-research/mdistiller>) and [Logit-Standarization KD](<https://github.com/sunshangquan/logit-standardization-KD>)


#### Key Features
- **Dynamic Sampling**: Sample selection based on Teacher-Student KL divergence
- **Penalty System**: Weight adjustment based on Teacher-Target divergence
- **Adaptive Learning**: Sampling ratio adjustment according to epochs

#### KDAS Configuration Parameters
```yaml
CFG.SOLVER.TRAINER = "kdas" # or "base" for vanilla method
...
KDAS:
  START_RATE: 0.6                # Initial sampling ratio
  END_RATE: 0.4                  # Final sampling ratio
  SAMPLING_PERIOD: [1,3,5,10]    # Sampling period
  EXCLUSION_RATE: 0.0            # Exclusion rate
  PENALTY_FACTOR: 0.5            # Minimum penalty weight
  PENALTY_LAMBDA: 1000           # Penalty strength
  PENALTY_WARMUP: 0              # Penalty warmup
  WARMUP_EPOCHS: 0               # Overall warmup
  THRESHOLD: 0.2                 # Penalty threshold
```

#### Installation

The implementation requires the following packages:
- torch
- yacs
- wandb
- scipy
- tqdm
- tensorboardX

Install the package:

```
sudo pip3 install -r requirements.txt
sudo python setup.py develop
```

### Distilling CNNs

- Download the [`cifar_teachers.tar`](<https://github.com/megvii-research/mdistiller/releases/tag/checkpoints>) and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

#### 1. For KD
```bash
# KD + KDAS
python tools/train.py --cfg configs/cifar100/kd/vgg13_vgg8.yaml
```

#### 2. For DKD
```bash
# DKD + KDAS
python tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml
```

#### 3. For LogitSTD
```bash
# LogitSTD + KDAS
python tools/train.py --cfg configs/cifar100/kd/vgg13_vgg8.yaml --logit-stand --base-temp 2 --kd-weight 9
```


## Citation

If you find that this project helps your research, please consider citing some of the following paper:

```
TBD
```
