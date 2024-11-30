# Tomato Leaf Disease Detection
deep CNN for image classification

## Introduction
This project aims to detect diseases on tomato leaves and recommend appropriate solutions.

## Dataset
The dataset is available at [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and contains 10 classes of tomato leaf diseases.

| Class | Train Samples | Test Samples |
|-------|---------------|--------------|
| Tomato___Bacterial_spot | 2127          | 532          |
| Tomato___Early_blight | 2008          | 502          |
| Tomato___Late_blight | 1908          | 381          |
| Tomato___Leaf_Mold | 958           | 191          |
| Tomato___Septoria_leaf_spot | 2127          | 425          |
| Tomato___Spider_mites Two-spotted_spider_mite | 1676          | 335          |
| Tomato___Target_Spot | 1404          | 280          |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 3280          | 656          |
| Tomato___Tomato_mosaic_virus | 373           | 77           |
| Tomato___healthy | 1591          | 318          |

## Technologies
- pytorch
- sklearn
- python

## training
If you want to train a model with a specific dataset, you could run:

- **python train_model.py --root your_dataset_root**

If you want to train a model with common dataset and your preference parameters, like optimizer and learning rate, you could run:

- **python train_model.py -e your_epoch -b batch_size**

If want to run my model, you could run:

- **python View.py**

## Experiment
My model is quite lightweight so I run directly on a local machine with an NVIDIA GeForce RTX 3060 GPU.

The training/testing accuracy/loss curve for each experiment of the dataset with more than 50 epochs is shown below:

- train
![z6075821734984_8bfb7df4ab7179d759fc3da814c901a0.jpg](image/z6075821734984_8bfb7df4ab7179d759fc3da814c901a0.jpg)

- test
![z6075821734983_5ec2b77f6369517c2d6644ba83779287.jpg](image/z6075821734983_5ec2b77f6369517c2d6644ba83779287.jpg)

- Confusion Matrix
![z6075821734966_4d41bc4e29e9d01ac790ec0937d9598a.jpg](image/z6075821734966_4d41bc4e29e9d01ac790ec0937d9598a.jpg)

- result

[![Video Demo](https://img.youtube.com/vi/CClI0O1bD9o/0.jpg)](https://youtu.be/CClI0O1bD9o)


