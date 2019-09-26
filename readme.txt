# Convolutional Auto-Encoders with A3C for Super Mario Bros.

The objective of this project is to test the generalization capabilities of reinforcement learning on unseen states. 
The baseline A3C is efficient but can only be trained on individual levels


## Design

Input from gym --> Component 1 --> Component 2 --> Output

Component 1
- Encoder Conv Layer 1 (weights frozen)
- Encoder Conv Layer 2 (weights frozen)

Component 2
- conv layer 1
- conv layer 2
- LSTM
- Actor & Critic

## Process:
1) Train Convolutiona Auto-Encoder (CAE) and save model and weights
2) Design CAE with A3C


## Requirements

gym==0.10.9
torchvision==0.4.0
scikit_image==0.15.0
matplotlib==3.1.1
numpy==1.17.0
torch==1.2.0
opencv_python==4.1.0.25
gym_super_mario_bros==7.2.3
nes_py==8.1.1
skimage==0.0
tensorboardX==1.8

-You need a GPU (CUDA) to run the project, the project will not work efficiently on CPU
-It also only on Python3

## Testing 

CAE trained models already provided, more information to follow soon

to run training process of Mario, run python3 main_learn.py
to test project, run python3 main_test.py
to run convolutional auto encoder, run python3 convolutional_autoencoder.py
to run sequential auto encoder, run python3 sequential_autoencoder.py
