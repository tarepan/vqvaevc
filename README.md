<div align="center">

# WaveRNN + VQ-VAE <!-- omit in toc -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]

</div>

Reimplmentation of mkotha/WaveRNN's VQ-VAE WaveRNN  

## Current status
Currently analyze repository and try to reproduce the results.  
Reproduction target: multi-speaker VQ-VAE ([VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651))  
Training time: 945k steps (from original repo's issue)  

### ToDo
1. Dry run without error
2. Meature training time 
3. Reproduce results

## System details
- Task: speech reconstruction and speaker conversion  
- Dataset: Multi-speaker (VCTK)

## Demo
[Audio samples](https://mkotha.github.io/WaveRNN/).

## Usage

### Preparation

#### Requirements

* Python 3.6 or newer
* PyTorch with CUDA enabled
* [librosa](https://github.com/librosa/librosa)
* [apex](https://github.com/NVIDIA/apex) if you want to use FP16 (it probably
  doesn't work that well).


#### Create config.py

```
cp config.py.example config.py
```

#### Preparing VCTK

1. Download and uncompress [the VCTK dataset](https://datashare.is.ed.ac.uk/handle/10283/2651).
2. `python preprocess_multispeaker.py /path/to/dataset/VCTK-Corpus/wav48/ path/to/output/directory`
3. In `config.py`, set `multi_speaker_data_path` to point to the output directory.

### Train

#### Quick training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][notebook]

#### CLI

```
$ python wavernn.py
```

Trained models are saved under the `model_checkpoints` directory.

By default, the script will take the latest snapshot and continues training from there.  
To train a new model freshly, use the `--scratch` option.  

Every 50k steps, the model is run to generate test audio outputs.  
The output goes under the `model_outputs` directory.  

When the `-g` option is given, the script produces the output using the saved model, rather than training it.  

## Training Speed <!-- omit in toc -->
x.xx [iter/sec] @ NVIDIA T4 Google Colaboratory (AMP+/-)

# Deviations from the papers

I deviated from the papers in some details, sometimes because I was lazy, and sometimes because I was unable to get good results without it.  
Below is a (probably incomplete) list of deviations.

All models:

* The sampling rate is 22.05kHz.

VQ-VAE:

* I normalize each latent embedding vector, so that it's on the unit 128-dimensional sphere.  
  Without this change, I was unable to get good utilization of the embedding vectors.
* In the early stage of training, I scale with a small number the penalty term that apply to the input to the VQ layer.  
  Without this, the input very often collapses into a degenerate distribution which always selects the same embedding vector.
* During training, the target audio signal (which is also the input signal) is
  translated along the time axis by a random amount, uniformly chosen from
  [-128, 127] samples. Less importantly, some additive and multiplicative
  Gaussian noise is also applied to each audio sample. Without these types of
  noise, the feature captured by the model tended to be very sensitive to small
  purterbations to the input, and the subjective quality of the model output
  kept descreasing after a certain point in training.
* The decoder is based on WaveRNN instead of WaveNet. See the next section for
  details about this network.

# Context stacks

The VQ-VAE implementation uses a WaveRNN-based decoder instead of a WaveNet-based decoder found in the paper.  
This is a WaveRNN network augmented with a context stack to extend the receptive field.  
This network is defined in `layers/overtone.py`.  

The network has 6 convolutions with stride 2 to generate 64x downsampled 'summary' of the waveform, and then 4 layers of upsampling RNNs, the last of which is the WaveRNN layer.  
It also has U-net-like skip connections that connect layers with the same operating frequency.  

# References
- VQ-VAE
  - ["Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937)
  - [official demo](https://avdnoord.github.io/homepage/vqvae/)
- WaveRNN
  - ["Efficient Neural Audio Synthesis"](https://arxiv.org/abs/1802.08435)

# Acknowledgement
The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN).  
mkotha/WaveRNN is based on [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN).  

[notebook]:https://colab.research.google.com/github/tarepan/vqvaevc/blob/main/vqvaevc.ipynb