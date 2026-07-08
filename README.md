# Mobile-Jepa
Mobile-JEPA is a novel highly efficient CNN adaptation of I-JEPA. Has been built to preserve I- JEPA’s partial computation efficiency without relying on sparse convolutions introducing a localized crop blocking mechanism integrated with a modernized VGG style backbone, effectively leveraging CNN inductive biases for low resource regimes.
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/mobile-jepa-architecture-cor.png?raw=true" width="100%">

# Project structure
```
.
├── LICENSE
├── README.md
├── masking_exaples # Folder containing plots of block masking both for Mobile-Jepa strategy and I-Jepa
│   ├── I-Jepa-style
│   └── Mobile-Jepa-style
├── report.pdf 
├── requirements.txt
├── scripts
│   ├── parameters-counter.py # Use to count the models trainable parameters
│   ├── train-knn.py # Script to lunch for knn validation
│   ├── train-linearprobe.py # Script to lunch for linear probe validation
│   └── train.py # Main script to train Mobile-Jepa and I-Jepa (ViT/Tiny)
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── Dataset.py  # Dataset and data loaders handling 
│   │   └── __init__.py
│   ├── models
│   │   ├── BackbonesCNN.py # Backbones models for Mobile-Jepa (legacy experimental models are also presented)
│   │   ├── Mobile_JEPA.py # Main Mobile-Jepa logic class with the flows and croping and backbones usage
│   │   ├── ViT.py # ViT I-Jepa style
│   │   └── __init__.py
│   ├── utils
│   │   ├── BlockMasking.py # Mobile-Jepa Crop based block sampling
│   │   ├── IJepaBlockMasking.py # I-Jepa Crop based block sampling
│   │   ├── Schedulers.py # WarmupCosineSchedule and CosineWDSchedule util class used in train.py
│   │   ├── __init__.py
│   │   └── config.py # Helper function used to load configs.json across the codebase
│   └── visualization # Scripts to run for plotting attention maps, block masking, features and 2D latent t-sne
│       ├── __init__.py
│       ├── attention_maps.display.py
│       ├── display_mask.py 
│       ├── features_display.py
│       └── plot_latent_space.py
└── training_results # Main experimental results did trough the project NOTE: round1 were an erly architecture test in the first phaces i kept it for legacy, hence you can skip it.
    ├── round1
    ├── round10
    ├── round11
    ├── round2
    ├── round3
    ├── round4
    ├── round5
    ├── round6
    ├── round7
    ├── round8
```

# How to launch the scripts

⚠️ **WARNING:** For training scripts or knn evaluations having a GPU CUDA compatible is higly recomended

1. Create a virtual enviroptment: ```python -m venv venv ```
2. Activate the enviroptment (UNIX):  ```source ./venv/bin/activate```
3. Install dependencies: ```pip install -r requirements.txt```
4. Impost `training_results/params.json` as you wish, or use one of the avaible in the rounds
5. Impost `IS_VIT_BASE = True` in your running script if you want to run I-Jepa, ```False``` otherwise. Ensure the ```Dataset.py``` has setted right `IS_VIT_BASE` and the transformation up/down scalling if you wish to change it or is required to execute a round.
6. Run the chosen script, for example for the main training script: ```python train.py```

For the main training rounds I did you can find the weights in the following cloud directory: https://drive.google.com/drive/folders/1ICbjsMIbRSIqEIkJa7PvKiJBhgBq9XiW?usp=sharing



# Block masking examples

## Mobile-Jepa
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/masking_exaples/Mobile-Jepa-style/masked_image_grid_round10-11.png?raw=true" width="100%">

## I-Jepa
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/masking_exaples/I-Jepa-style/masked_image_grid-round9.png?raw=true" width="100%">

# Mobile-Jepa feature maps on ImangeNet-100 showcase

## Layer 1
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round11/features/Layer_1_inc.png?raw=true" width="100%">

## Layer 2
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round11/features/Layer_2_down1.png?raw=true" width="100%">

## Layer 3
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round11/features/Layer_3_down2.png?raw=true" width="100%">

## Layer 4
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round11/features/Layer_4_down3.png?raw=true" width="100%">

## Bootleneck
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round11/features/Layer_5_Bottleneck.png?raw=true" width="100%">

# I-Jepa on Vit/Tiny mean attention maps

## Part 1
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round9/attention_maps/vit_attention_mean_attention_map_part_1.png?raw=true" width="100%">

## Part 2
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round9/attention_maps/vit_attention_mean_attention_map_part_2.png?raw=true" width="100%">

## Part 3
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round9/attention_maps/vit_attention_mean_attention_map_part_3.png?raw=true" width="100%">

## Part 4
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round9/attention_maps/vit_attention_mean_attention_map_part_4.png?raw=true" width="100%">

# Mobile-Jepa: 2D t-SNE laten space with semantic related class and orthogonal class
<img src="https://github.com/PasqualeCelani/Mobile-Jepa/blob/main/training_results/round11/2dlaten-t-sne.drawio.png?raw=true" width="100%">

# Acknowledgements
I want to give credit to the original I-Jepa repository: https://github.com/facebookresearch/ijepa. I took inspiration and adapted the code of the code suchs as: ```BlockMasking.py (I tooke inspiration of the setup, and i adapted to my crop strategy)``` ```IJepaBlockMasking.py (as the original)```, ```Schedulers.py (adapted for checkpoint logic)```,  and how to setup and impost the the training loop. 

