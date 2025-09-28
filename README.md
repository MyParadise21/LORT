# LORT
(Speech Communication) LORT: Locally Refined Convolution and Taylor Transformer for Monaural Speech Enhancement

**Abstract:** 
Achieving superior enhancement performance while maintaining a low parameter count and computational complexity remains a challenge in the field of speech enhancement. In this paper, we introduce LORT, a novel architecture that integrates spatial-channel enhanced Taylor Transformer and locally refined convolution for efficient and robust speech enhancement. We propose a Taylor multi-head self-attention (T-MSA) module enhanced with spatial-channel enhancement attention (SCEA), designed to facilitate inter-channel information exchange and alleviate the spatial attention limitations inherent in Taylor-based Transformers. To complement global modeling, we further present a locally refined convolution (LRC) block that integrates convolutional feed-forward layers, time-frequency dense local convolutions, and gated units to capture fine-grained local details. Built upon a U-Net-like encoder-decoder structure with only 16 output channels in the encoder, LORT processes noisy inputs through multi-resolution T-MSA modules using alternating downsampling and upsampling operations. The enhanced magnitude and phase spectra are decoded independently and optimized through a composite loss function that jointly considers magnitude, complex, phase, discriminator, and consistency objectives. Experimental results on the VCTK+DEMAND and DNS Challenge datasets demonstrate that LORT achieves competitive or superior performance to state-of-the-art (SOTA) models with only 0.96M parameters, highlighting its effectiveness for real-world speech enhancement applications with limited computational resources.

## Pre-requisites
1. Python >= 3.8.
2. Clone this repository.
3. Install python requirements. Please refer requirements.txt.
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942).

## Training
For single GPU (Recommend), LORT needs at least 12GB GPU memery.
```
python train_lort.py
```

## Training with your own data
### Step 1: Generate dataset list files
To prepare data for model training, first run make_file_list.py.
```
python make_file_list.py
```
### Step 2: Start training with train-lort.py
After getting the list files (e.g., train.txt for training data, validation.txt for validation data), specify the file paths via command-line parameters to start training.
```
python train_lort.py --input_training_file ./train.txt --input_validation_file ./validation.txt
```
## Inference
```
python inference.py --checkpoint_file /PATH/TO/YOUR/CHECK_POINT/g_xxxxxxx
```

## Acknowledgements
We referred to [MP-SENet](https://github.com/yxlu-0102/MP-SENet), [MUSE](https://github.com/huaidanquede/MUSE-Speech-Enhancement)
