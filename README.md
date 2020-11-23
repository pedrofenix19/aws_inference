# Training and deploying GANs models in a cost effective way using AWS Sagemaker and Lambdas.

This repo contains the code used to train and deploy a Generative Adversarial Model (GAN) that generates digits using AWS Sagemaker Spot Instances and AWS Lambdas. The code was used to perform a DEMO in the "Sagemaker y Lambda: Entrenando y desplegando modelos generativos con coste eficiente" talk at the **Next Cloud Week 2020**

The basic idea is to use AWS Sagemaker Spot Instances to train a GAN with the [MNIST digits dataset](http://yann.lecun.com/exdb/mnist/) and then deploy the model to be accesible through a Lambda using the [Serverless](https://www.serverless.com/) Framework and the Tensorflow Lite library.

## Requirements
- An AWS account with the credential configured in the .aws/credentials file.
- AWS CLI
- Serverless Framework installed
- Docker

**IMPORTANT**: You need a Linux machine for this, I tried to make it work on WSL2 but got an error while compiling the Tensorflow Lite library on a Docker container. I have not tried it on a Mac so feel free to try it.

## Training the model

To train the model we need to create an AWS Sagemaker notebook instance. To do this open your AWS console and go to **Sagemaker** -> **Notebook** -> **Notebook instances** and create a notebook instance. Wait until it is create and open it. When the Jupyter environment opens, upload the files inside the **notebook_code** folder and open the **MNIST_GAN_Spot_Training** notebook. This notebook will use the **mnist_gan.py** script to train the model and will create a model.tflite file (The notebook itself has more details on what it does under the hood).

