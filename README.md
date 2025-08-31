#🌈 Colorization with UNet

This project implements a **deep learning model for automatic colorization of black-and-white images** using a **UNet-based architecture**. It leverages a dataset of paired color and grayscale images (from Kaggle) to learn how to predict color information from grayscale inputs.


---

## Features

- Convert black-and-white images to color.
- UNet architecture for high-quality colorization.
- Training pipeline with PyTorch.
- Inference script to colorize images from any directory.
- Save checkpoints and sample outputs during training.

## Dataset

The project expects a dataset structured as follows:
data/  
├── train_color/  
├── train_black/  
├── test_color/  
├── test_black/  

Images should be `.jpg`, `.jpeg`, or `.png`. The dataset used can be downloaded from [Kaggle](https://www.kaggle.com/).


## Project Structure

src/  
├── data.py       # dataset loader  
├── model.py      # UNet model definition  
├── utils.py      # helperss   
train.py          # training   
infer.py          # inference   
checkpoints/      # saved models  
samples/          # output   

##  Example Output
