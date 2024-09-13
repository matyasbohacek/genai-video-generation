# GenAI Video Generation

This repository contains open-source text-to-video model wrappers, simplifying their inference to a single, standardized function.

## Models

| Engine                                                                                                  | Example #1                                                                                          | Example #2                                                                                          | Example #3                                                                                          |
|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [ByteDance AnimateDiff Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)               | ![Example Video 3](https://github.com/user-attachments/assets/9626c33f-e58e-421e-aea8-97ad7068cf45) | ![Example Video x](https://github.com/user-attachments/assets/1ab0435a-fc99-4920-8711-a0103ef1c377) | ![Example Video x](https://github.com/user-attachments/assets/3ee4824e-fbfb-4177-91f8-63fbca1c9426) |
| [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)                                               | ![Example Video 1](https://github.com/user-attachments/assets/0f7b025a-0301-46be-beff-212667dde5b4) | ![Example Video x](https://github.com/user-attachments/assets/6c2a6627-3b83-4bb0-8d0c-cef9fe1456ef) | ![Example Video x](https://github.com/user-attachments/assets/232be060-1b5b-4883-a698-c5c2ea419ee4) |
| [Stable Diffusion Txt2Img + Img2Vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) | ![Example Video 2](https://github.com/user-attachments/assets/4a80889d-0e51-4129-87e2-764fc0ee321c) | ![Example Video x](https://github.com/user-attachments/assets/3e42a70a-8d2e-4681-9553-bcd6f206cf32) | ![Example Video x](https://github.com/user-attachments/assets/47fec49d-836b-43e3-9eb9-1f5d752242ff) |

## Deployment

Two deployment options are supported and documented: local deployment and Colab deployment. To get started and choose the right deployment configuration, consider your compute resources. If you have an NVIDIA GPU with large memory and recent CUDA version (A100/H100 or better), you should be able to deploy this repository locally. Else, opt for a remote environment with those resourches, such as Google Colab. Note that you need to be subscribed to Colab+ to get A100s.

### Local Deployment

1. Create a Conda enviornment:  `conda create -n txt2vid python=3.10`
2. Install required packages: `pip install -r requirements.txt`.
3. Log into your Hugging Face: `huggingface-cli login`.

### Colab Deployment

Open [this Colab]() to get started; instructions are included. 
