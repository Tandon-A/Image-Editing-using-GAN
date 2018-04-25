import tensorflow as tf
"""
Import helper functions
"""
from layers import *


"""
Class Definition of CycleGAN as presented in paper by Zhu et. al.
"""

class CycleGAN:
    
    def __init__(self,batch_size,input_shape,pool_size,beta1):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.lr_rate = tf.placeholder(dtype=tf.float32,shape=[],name="lr_rate")
        self.input_A,self.input_B, self.fake_pool_Aimg, self.fake_pool_Bimg = self.model_inputs(self.batch_size,self.input_shape[0],self.input_shape[1],self.input_shape[2])
        
        self.gen_A,self.gen_B,self.cyclicA,self.cyclicB,self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B = self.model_arc(self.input_A,self.input_B,self.fake_pool_Aimg,self.fake_pool_Bimg)
        
        
        self.gen_loss_A, self.disc_loss_A, self.gen_loss_B,self.disc_loss_B = self.model_loss(self.real_disc_A,self.real_disc_B,self.fake_disc_A,self.fake_disc_B,self.fake_pool_disc_A,self.fake_pool_disc_B,self.input_A,self.cyclicA,self.input_B,self.cyclicB)
        
        
        self.discA_opt,self.discB_opt,self.genA_opt,self.genB_opt = self.model_opti(self.gen_loss_A,self.disc_loss_A,self.gen_loss_B,self.disc_loss_B,self.lr_rate,beta1)
        
        
    """
    Function to model inputs of the network
    """
    def model_inputs(self,batch_size,height,width,channels):
        input_A = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="input_A")
        input_B = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="input_B")         
        gen_pool_A = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="fake_pool_Aimg") 
        gen_pool_B = tf.placeholder(dtype=tf.float32,shape=[batch_size,width,height,channels],name="fake_pool_Bimg") 
        
        return input_A,input_B,gen_pool_A,gen_pool_B
    
    """
    Function to model architecture of CycleGAN. 
    """
    def model_arc(self,input_A,input_B,fake_pool_A,fake_pool_B):
        
        with tf.variable_scope("CycleGAN") as scope:
            gen_B = generator(input_A,name="generator_A")
            gen_A = generator(input_B,name="generator_B")
            real_disc_A = discriminator(input_A,name="discriminator_A")
            real_disc_B = discriminator(input_B,name="discriminator_B")
            
            scope.reuse_variables()
            
            fake_disc_A = discriminator(gen_A,name="discriminator_A")
            fake_disc_B = discriminator(gen_B,name="discriminator_B")
            
            cyclicA = generator(gen_B,name="generator_B")
            cyclicB = generator(gen_A,name="generator_A")
            
            scope.reuse_variables()
            
            fake_pool_disc_A = discriminator(fake_pool_A,name="discriminator_A")
            fake_pool_disc_B = discriminator(fake_pool_B,name="discriminator_B")
            
            return gen_A,gen_B,cyclicA,cyclicB,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B
        
    
    """
    Function to calculate loss.
    """    
    def model_loss(self,real_disc_A,real_disc_B,fake_disc_A,fake_disc_B,fake_pool_disc_A,fake_pool_disc_B,input_A,cyclicA,input_B,cyclicB):
        
        cyclic_loss = tf.reduce_mean(tf.abs(input_A-cyclicA)) + tf.reduce_mean(tf.abs(input_B - cyclicB))
        
        disc_loss_A = tf.reduce_mean(tf.squared_difference(fake_disc_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(fake_disc_B,1))
        
        gen_loss_A = cyclic_loss*10 + disc_loss_B
        gen_loss_B = cyclic_loss*10 + disc_loss_A
        
        d_loss_A = (tf.reduce_mean(tf.square(fake_pool_disc_A)) + tf.reduce_mean(tf.squared_difference(real_disc_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(fake_pool_disc_B)) + tf.reduce_mean(tf.squared_difference(real_disc_B,1)))/2.0

        return gen_loss_A,d_loss_A,gen_loss_B,d_loss_B
    
    """
    Function to optimize the network. Used Adam Optimizer.
    """
    def model_opti(self,gen_loss_A,disc_loss_A,gen_loss_B,disc_loss_B,lr_rate,beta1):
        
        train_vars = tf.trainable_variables()
        discA_vars = [var for var in train_vars if var.name.startswith('CycleGAN/discriminator_A')]
        discB_vars = [var for var in train_vars if var.name.startswith('CycleGAN/discriminator_B')]
        genA_vars = [var for var in train_vars if var.name.startswith('CycleGAN/generator_A')]        
        genB_vars = [var for var in train_vars if var.name.startswith('CycleGAN/generator_B')]        
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            discA_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_A,var_list = discA_vars)
            discB_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss_B,var_list = discB_vars)
            genA_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_A,var_list = genA_vars)
            genB_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss_B,var_list = genB_vars)
    
        return discA_train_opt, discB_train_opt, genA_train_opt, genB_train_opt
        
    
    
    
        
        
        
        
