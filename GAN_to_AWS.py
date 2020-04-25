from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
from tensorflow import keras
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
from keras import backend

channel=1
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
dataset_HR = train_images/255
dataset_HR=np.reshape(dataset_HR,[60000,28,28,1])

BUFFER_SIZE = 60000
BATCH_SIZE = 256
downscale=4
dataset_LR =np.zeros([np.shape(dataset_HR )[0],int(np.shape(dataset_HR )[1]/downscale),int(np.shape(dataset_HR )[2]/downscale),channel])

x=np.linspace(0, np.shape(dataset_HR )[1]-downscale, int(np.shape(dataset_HR )[1]/downscale))
# 下采样图
for i in range (np.shape(dataset_LR )[0]):
    for j in range (np.shape(dataset_LR )[1]):
        a=int(x[j])
        for k in range (np.shape(dataset_LR )[1]):
            b=int(x[k])
            dataset_LR[i][j][k]=dataset_HR[i][a][b]


# print(np.shape(dataset_HR),np.shape(dataset_LR))
# print( 'kiya')

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
        
    model = keras.layers.add([gen, model])   
    return model



def discriminator_block(model, filters, kernel_size, strides):
    
    model = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = keras.layers.LeakyReLU(alpha = 0.2)(model)
    
    return model
def generator(width, height, upscale):
    
    rate=upscale*upscale
    # 上采样
    input_pic=keras.layers.Input(shape=(width, height,1),name='input_picture')
    z=keras.layers.Conv2D(rate,3, strides=1, padding='same',use_bias=False)(input_pic)
    upscale_width=width*upscale
    upscale_height=height*upscale
    #z=keras.layers.Reshape((upscale_width,upscale_height，1))(z)
    
    z=keras.layers.Reshape((upscale_width,upscale_height,1))(z)
    
    model=z   
            # Using 5 Residual Blocks
    for index in range(5):
        model = res_block_gen(model, 3, 64, 1)
	    
    model = keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = keras.layers.BatchNormalization(momentum = 0.5)(model)
    model = keras.layers.add([z, model])
	    
    model = keras.layers.Conv2D(filters = 1, kernel_size = 9, strides = 1, padding = "same")(model)
    #model = keras.layers.Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model) #三通道
    model = keras.layers.Activation('tanh')(model)
	   
    generator_model = keras.Model(inputs = input_pic, outputs = model)
        
    return generator_model
    

def discriminator(width, height):      
    dis_input = keras.layers.Input(shape=((width), (height),1),name='input_picture')
        
    model = keras.layers.Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
    model = keras.layers.LeakyReLU(alpha = 0.2)(model)
        
    model = discriminator_block(model, 64, 3, 2)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 2)
    model = discriminator_block(model, 256, 3, 1)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 512, 3, 1)
    model = discriminator_block(model, 512, 3, 2)
        
    model = keras.layers.Flatten()(model)
    model = keras.layers.Dense(1024)(model)
    model = keras.layers.LeakyReLU(alpha = 0.2)(model)
       
    model = keras.layers.Dense(1)(model)
    model = keras.layers.Activation('sigmoid')(model) 
        
    discriminator_model = keras.Model(inputs = dis_input, outputs = model)
       
    return discriminator_model

def mean_squared_error(y_true, y_pred):
    return sum(sum(sum(backend.mean(keras.backend.square(y_pred - y_true), axis=-1))))


def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = keras.layers.Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = keras.Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def train_step(data_LR,data_HR,batch_size,batch_num,width,height,upscale):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        model_gen = generator(width,height,upscale)
        model_disc=discriminator(width*4,height*4)

        gen_LR=dataset_LR[batch_num*batch_size:(batch_num+1)*batch_size,:,:,:]
        real_HR=dataset_HR[batch_num*batch_size:(batch_num+1)*batch_size,:,:,:]     
    
        generated_raw = model_gen(gen_LR, training=True)
    
        real_output = model_disc(generated_raw, training=True)
        gen_output = model_disc(generated_raw, training=True)

        gen_loss = (mean_squared_error(generated_raw,real_HR))/batch_size
        disc_loss = discriminator_loss(real_output, gen_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, model_gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model_disc.trainable_variables)
    
def generate_and_save_images(model, epoch, test_input):
  # 注意 training` 设定为 False
  # 因此，所有层都在推理模式下运行（batchnorm）。
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(28,28))

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()    

def train(dataset_LR,dataset_HR, epochs,batch_size, width, height, upscale):
    batch_count=np.shape(dataset_HR)[0]/batch_size
    for epoch in range(epochs):
        print('epoch',epoch,'begin')
        start = time.time()
        for batch_number in range (int(batch_count)):
            print('processing',batch_number)
            train_step(dataset_LR,dataset_HR,batch_size,batch_number,width,height,upscale)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        if (epoch + 1) % 5 == 1:
            generator.save('./training_checkpoints/gen_model%d.hdf5' % e)
            discriminator.save('./training_checkpoints/dis_model%d.hdf5' % e)
        generate_and_save_images(generator, epoch, dataset_LR[0:1,:,:,:])
        
train(dataset_LR,dataset_HR, 10,300, 7, 7, 4)
