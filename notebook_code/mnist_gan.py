#This code is based on the following link
#https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/

#Modifications were done so it can resume from a partially trained GAN

import argparse
import os
import numpy as np
import re
import random
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Reshape
from keras.layers import Conv2D, Conv2DTranspose
import sys
import tensorflow as tf

def define_discriminator(learning_rate = 0.0002):
    '''
        This function defines the discriminator model and compiles it
    '''
    in_shape=(28,28,1)
    
    if not LAST_SAVED_EPOCH:
        print(f"There is no saved model. We build the model from scratch.")
        model = Sequential()
        model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
    else:
        model = load_model(os.path.join(os.path.join(*[CHECKPOINTS_PATH, "discriminator",f"d_model_{LAST_SAVED_EPOCH:03d}.h5"])))
    
    #compile model
    opt = Adam(lr=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    '''
        This function defines the generator model. It is not compiled since it will be part 
        of the greater GAN model and will not be trained alone
    '''
    
    if not LAST_SAVED_EPOCH:
        print(f"There is no saved model. We build the model from scratch.")
        model = Sequential()

        #foundation for the 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7,7,128)))

        #upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        #upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    else:
        model = load_model(os.path.join(os.path.join(*[CHECKPOINTS_PATH, "generator",f"g_model_{LAST_SAVED_EPOCH:03d}.h5"])))
    
    return model

def define_gan(g_model, d_model, learning_rate = 0.0002):
    '''
        We define the GAN as the discriminator model attached to the generator 
    '''
    d_model.trainable = False
    
    model = Sequential()
    
    model.add(g_model)
    model.add(d_model)
    
    opt = Adam(lr=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def smooth_positive_labels(y):
    '''
        We are applying label smoothing as a regularization technique
    '''
    return y - 0.3 + (np.random.random(y.shape) * 0.5) 

def prepare_dataset(dataset):
    '''
        We apply normalization to our images.
    '''
    X = np.expand_dims(dataset, axis=-1)
    X = X.astype('float32')
    X = X / 255.0
    return X

def generate_real_samples(dataset, n_samples):
    '''
        This function draws real examples from the dataset
    '''
    #choose random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    
    #retrieve selected images
    X = dataset[ix]
    
    #generate 'real' class label (1)
    y = smooth_positive_labels(np.ones((n_samples, 1)))
    
    return X, y

def generate_fake_samples(g_model, latent_dim, n_samples):
    '''
        This function generates fake samples by passing 
        random noise to the generator model
    '''
    x_input = generate_latent_points(latent_dim, n_samples)
    
    X = g_model.predict(x_input)
    
    y = np.zeros((n_samples, 1))
    
    return X, y

def train_discriminator(model, dataset, n_iter=100, n_batch=256):
    '''
        This function trains the discriminator model. 
        It feeds the model with half real images and half fake 
        images (generated by the generator model)
    '''
    half_batch = int(n_batch/2)
    
    for i in range(n_iter):
        x_real, y_real = generate_real_samples(dataset, half_batch)
        _, real_acc = model.train_on_batch(x_real, y_real)
        
        x_fake, y_fake = generate_fake_samples(half_batch)
        _, fake_acc = model.train_on_batch(x_fake, y_fake)
        
        print(f">{i:4d}: real_acc: {real_acc:.4f}, fake_acc: {real_acc:.4f}")
        
def generate_latent_points(latent_dim, n_samples):
    '''
        This function generates the latent points that will be 
        used as input for the generator model
    '''
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input
        
def save_plot(examples, epoch, n=10):
    '''
        This functions saves the example images generated by
        the generator model as it progresses its training
    '''
    fig, ax = plt.subplots(n, n)
    for i in range(n):
        for j in range(n):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(examples[random.randint(0,examples.shape[0] - 1),:,:,0], cmap='gray_r')
            
    filename = f"generated_plot_e{epoch:04d}.png"
    if not os.path.exists(os.path.join(*[CHECKPOINTS_PATH, "plots"])):
        os.makedirs(os.path.join(*[CHECKPOINTS_PATH, "plots"]))
    fig.savefig(os.path.join(*[CHECKPOINTS_PATH, "plots", filename]))
    plt.close()
        
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    '''
        This function evaluates the accuracy of the discriminator model at a
        given point in the GAN training.
    '''
    x_real, y_real = generate_real_samples(dataset, n_samples)
    
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    
    print(f">Accuracy real: {acc_real:.4f}, fake: {acc_fake:.4f}")
        
    save_plot(x_fake, epoch)
    
        
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    '''
        This function performs the GAN training
    '''
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch/2)
    
    #manually enumerate epochs
    for i in range(LAST_SAVED_EPOCH + 1, n_epochs + 1):
        
        #enumerate batches over the training set
        for j in range(bat_per_epo):
            #get randomly selected 'real' samples
            x_real, y_real = generate_real_samples(dataset, half_batch)
            
            #generate 'fake' examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            
            #create training set for the discriminator
            X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            
            #update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            
            #prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            
            #create inverted labels for the fake samples
            y_gan = smooth_positive_labels(np.ones((n_batch, 1)))
            
            #update generator model via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            
            print(f"Epoch: {i:4d}/{n_epochs}, batch: {j+1:4d}/{bat_per_epo}, d={d_loss:.4f}, g={g_loss:.4f}")
            
        if (i) % 2 == 0:
            save_models(g_model, d_model, i)
            summarize_performance(i, g_model, d_model, dataset, latent_dim)
            
def save_models(g_model, d_model, epoch):
    '''
        This function saves the models at a given point in the GAN training.
        The models are saved in the CHECKPOINTS_PATH folder so AWS would save
        them whenever it wants to destroy the instance in which the algorithm is running
    '''
    if not os.path.exists(os.path.join(*[CHECKPOINTS_PATH, "generator"])):
        os.makedirs(os.path.join(*[CHECKPOINTS_PATH, "generator"]))
        
    g_model.save(os.path.join(*[CHECKPOINTS_PATH, "generator", f"g_model_{epoch:03d}.h5"]))
    
    if not os.path.exists(os.path.join(*[CHECKPOINTS_PATH, "discriminator"])):
        os.makedirs(os.path.join(*[CHECKPOINTS_PATH, "discriminator"]))
        
    g_model.save(os.path.join(*[CHECKPOINTS_PATH, "discriminator", f"d_model_{epoch:03d}.h5"]))
    
def get_last_saved_epoch():
    '''
        If the training was previously interrupted by AWS, this function recovers the last epoch 
        in which the models were saved so we can resume training from there
    '''
    if not os.path.exists(os.path.join(*[CHECKPOINTS_PATH, "generator"])):
        return 0
    
    g_files = [f for f in os.listdir(os.path.join(*[CHECKPOINTS_PATH, "generator"])) if f.endswith('.' + 'h5')]
    d_files = [f for f in os.listdir(os.path.join(*[CHECKPOINTS_PATH, "discriminator"])) if f.endswith('.' + 'h5')]
    
    g_epoch = max([int(re.search(r'g_model_(.*?)\.',f).group(1)) for f in g_files])
    d_epoch = max([int(re.search(r'd_model_(.*?)\.',f).group(1)) for f in d_files])
    
    return min([g_epoch, d_epoch])
            
            
if __name__ == '__main__':
    
    #The arguments are passed by Sagemaker to the training script
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--job_name', type=str, default="mnist-gan")
    parser.add_argument('--gpu_count', type=int, default=os.environ['SM_NUM_GPUS'])
    
    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--checkpoints_path',  type=str,   default='/opt/ml/checkpoints')
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    args, _ = parser.parse_known_args()
    
    epochs = args.epochs
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    job_name = args.job_name
    CHECKPOINTS_PATH = os.path.join(*[args.checkpoints_path])
    GPU_COUNT = args.gpu_count
    
    #We load the dataset and the last saved epoch in case the training was
    #previously stopped
    dataset = np.load(os.path.join(*[args.train, 'mnist.npy']))
    
    LAST_SAVED_EPOCH = get_last_saved_epoch()
    
    
    d_model = define_discriminator(learning_rate)
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model, learning_rate)
    dataset = prepare_dataset(dataset)
    train(g_model, d_model, gan_model, dataset, latent_dim,
        n_epochs=epochs, n_batch=batch_size)
    
    g_model.save(os.path.join(*["/opt/ml/model","g_final_model.h5"]))
