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
import random

from IPython import display
from tensorflow.keras import backend

from skimage.metrics import mean_squared_error
#from skimage.metrics import compare_mse

channel=1
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# dataset_HR = train_images/255
# dataset_HR=np.reshape(dataset_HR,[60000,28,28,1])

dataset_HR = train_images[0:600,:,:]/255
dataset_HR=np.reshape(dataset_HR,[600,28,28,1])
#ones=np.ones(60000)
ones=np.ones(600)



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
    for index in range(35):
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


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(5e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
GAN_optimizer = tf.keras.optimizers.Adam(1e-4)

def get_gan_network(width, height,upscale,generator,discriminator, generator_optimizer,discriminator_optimizer):
    
    discriminator.trainable = False
    gan_input = keras.layers.Input(shape=((width), (height),1))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = keras.Model(inputs=gan_input, outputs=[x,gan_output])
                                   
                                   
    gan.compile(loss=['mean_squared_error', 'binary_crossentropy'],#discriminator_loss],
                loss_weights=[1., 1e-3],
                optimizer=GAN_optimizer,metrics=['mse','accuracy'])
    return gan
 
def train(epochs, batch_size,width, height,upscale,generator_optimizer,discriminator_optimizer,dataset_LR,dataset_HR):
    plt.imshow(dataset_HR[1,:,:,0] * 255, cmap='gray')
    plt.savefig('raw.png')

    generator_model = generator(width, height,4)
                                   
    discriminator_model = discriminator(width*4, height*4)


    generator_model.compile(loss='mean_squared_error', optimizer=generator_optimizer)
    discriminator_model.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer,metrics=['accuracy'])
    
    gan = get_gan_network(width, height,upscale,generator_model,discriminator_model, generator_optimizer,discriminator_optimizer)
    #gan.compile(loss=[mean_squared_error, discriminator_loss],optimizer=[generator_optimizer,discriminator_optimizer])

    batch_count=int(np.shape(dataset_LR)[0]/batch_size)                                 
    
    label=np.zeros(2*batch_size)
    label[batch_size+1:2*batch_size]=1
    label_ones=np.ones(batch_size)
    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for i in range(batch_count):
            print('batch',i)

            data_LR = dataset_LR[i*batch_size:(i+1)*batch_size,:,:,:]
            data_SR = generator_model.predict(data_LR)
            data_HR = dataset_HR[i*batch_size:(i+1)*batch_size,:,:,:]
            
            # print('sum_LR',np.sum(data_LR))
            # print('sum_HR',np.sum(data_HR))

            discriminator_model.trainable = True
            generator_model.trainable = False

            data=np.concatenate((data_SR,data_HR),axis = 0)                                
            #data_new=                     
            disc_loss = discriminator_model.train_on_batch(data, label)
            print(discriminator_model.metrics_names)
            print('disc_loss',disc_loss)
            #print('data_shape',np.shape(data))

            discriminator_model.trainable = False
            generator_model.trainable = True

            for j in range (30):
            #loss_gen = generator_model.train_on_batch(data_LR,data_HR)
              loss_gen = gan.train_on_batch(data_LR,[data_HR,label_ones])
            print(gan.metrics_names)
            #print('shape_loss_gen',np.shape(loss_gen))
            print('loss_gen',loss_gen)
        loss = gan.evaluate(dataset_LR, [dataset_HR,ones])

        SR_show=generator_model.predict(dataset_LR[1:2,:,:,:])
        plt.imshow(SR_show[0,:,:,0] * 255, cmap='gray')      
        plt.savefig("%d.jpg"%(e))
        
        print(gan.metrics_names)
        print('epoch',e,loss)


        # if e == 1 or e % 15 == 0:
#             plot_generated_images(e, generator)
        if e % 10 == 1:
             generator_model.save('gen_model.hdf5')
             discriminator_model.save('dis_model.hdf5')
            gan.save('gan_model.hdf5')

train(100,300,7,7,4,generator_optimizer,discriminator_optimizer,dataset_LR,dataset_HR)
