import tensorflow as tf 
import numpy as np 


"""
Class Definition of DCGAN as presented in paper by Radford et. al. 
"""

class DCGAN:
    
    def __init__(self,input_shape,z_dim,lr_rate,beta1=0.5,alpha=0.2):
        self.lr_rate = tf.Variable(lr_rate,trainable=False)
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.input_real,self.input_z = self.model_io(self.input_shape,self.z_dim)
        
        self.disc_loss,self.gen_loss,self.output_gen = self.model_loss(self.input_real,self.input_z,self.input_shape[2])        
        self.disc_opt,self.gen_opt = self.model_opti(self.disc_loss,self.gen_loss,self.lr_rate,beta1)

     
   
"""    
Function to model placeholders for input of both generator and discriminator
"""     
    def model_io(self,input_shape,z_dim):
        inputs_real = tf.placeholder(tf.float32,shape=(None,input_shape[0],input_shape[1],input_shape[2]),name="input_real")
        inputs_z = tf.placeholder(tf.float32,shape=(None,z_dim),name="input_z")
        return inputs_real,inputs_z   



"""
Generator 
"""
    def generator(self,z,out_channel_dim,is_train=True):
        w_init = tf.random_normal_initializer(mean=0.0, stddev = 0.02)
        with tf.variable_scope('generator',reuse= False if is_train==True else True):
            
            x1 = tf.reshape(z,(-1,1,1,100))
            
            deconv1 = tf.layers.conv2d_transpose(x1,512,4,1,padding='VALID',kernel_initializer=w_init)
            bn_norm1 = tf.layers.batch_normalization(deconv1,training=is_train)
            relu1 = tf.nn.relu(bn_norm1)
            
            deconv2 = tf.layers.conv2d_transpose(relu1,256,4,2,padding='SAME',kernel_initializer=w_init)
            bn_norm2 = tf.layers.batch_normalization(deconv2,training=is_train)
            relu2 = tf.nn.relu(bn_norm2)
            
            deconv3 = tf.layers.conv2d_transpose(relu2,128,4,2,padding='SAME',kernel_initializer=w_init)
            bn_norm3 = tf.layers.batch_normalization(deconv3,training=is_train)
            relu3 = tf.nn.relu(bn_norm3)
            
            deconv4 = tf.layers.conv2d_transpose(relu3,64,4,2,padding='SAME',kernel_initializer=w_init)
            bn_norm4 = tf.layers.batch_normalization(deconv4,training=is_train)
            relu4 = tf.nn.relu(bn_norm4)
            
            deconv5 = tf.layers.conv2d_transpose(relu4,out_channel_dim,4,2,padding='SAME',kernel_initializer=w_init)
            
            logits = deconv5
            out = tf.tanh(deconv5,name="generator_out")
            
            return out,logits  

  

    
"""
Discriminator
"""
    def discriminator(self,images,alpha=0.2,reuse = False):
       w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
       with tf.variable_scope('discriminator',reuse=reuse):
           
           conv1 = tf.layers.conv2d(images,64,4,2,padding='SAME',kernel_initializer=w_init)
           lrelu1 = tf.nn.leaky_relu(conv1,alpha=alpha)
           
           conv2 = tf.layers.conv2d(lrelu1,128,4,2,padding='SAME',kernel_initializer=w_init)
           bn_norm1 = tf.layers.batch_normalization(conv2,training=True)
           lrelu2 = tf.nn.leaky_relu(bn_norm1,alpha=alpha)
           
           conv3 = tf.layers.conv2d(lrelu2,256,4,2,padding='SAME',kernel_initializer=w_init)
           bn_norm2 = tf.layers.batch_normalization(conv3,training=True)
           lrelu3 = tf.nn.leaky_relu(bn_norm2,alpha=alpha)
           
           conv4 = tf.layers.conv2d(lrelu3,512,4,2,padding='SAME',kernel_initializer=w_init)
           bn_norm3 = tf.layers.batch_normalization(conv4,training=True)
           lrelu4 = tf.nn.leaky_relu(bn_norm3,alpha=alpha)
           
           conv5 = tf.layers.conv2d(lrelu4,512,4,2,padding='SAME',kernel_initializer=w_init)
           
           flat = tf.reshape(conv5,(-1,512))
           
           logits = tf.layers.dense(flat,1,kernel_initializer=w_init)
           
           out = tf.sigmoid(logits,name="discriminator_out")
           
           return out,logits


"""
Function to model the loss for both generator and discriminator. Returns the generated image.
"""       
    def model_loss(self,input_real,input_z,out_channel_dim):
        
	"""
	Images from the dataset are labelled with label = 0.9 for better training
	"""	
        label_smooth = 0.9 
        
        #get output of generator
        gen_img, gen_logits = self.generator(input_z,out_channel_dim,True)

	#pass real image to dicriminator
        disc_model_real, disc_logits_real = self.discriminator(input_real)
	
	#pass generated image to dicriminator
        disc_model_fake, disc_logits_fake = self.discriminator(gen_img,reuse=True)
            
	   	
        disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_real,labels=label_smooth*tf.ones_like(disc_model_real))) 
        disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,labels=tf.zeros_like(disc_model_fake)))
        

	"""
	Loss for discriminator is sum of loss for real image and fake image 
	"""	
        disc_loss = disc_loss_real + disc_loss_fake
        

        """
	To find loss for generator, fake image is passed with label= real (0.9)
	"""
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_logits_fake,labels=label_smooth*tf.ones_like(disc_model_fake)))
        
        return disc_loss,gen_loss,gen_img



    """
    Train generator and dicriminator using Adam Optimizer
    """ 
    def model_opti(self,disc_loss,gen_loss,lr_rate,beta1):
        
        train_vars = tf.trainable_variables()

        disc_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            disc_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(disc_loss,var_list = disc_vars)
            gen_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(gen_loss,var_list=gen_vars)
        
        return disc_train_opt, gen_train_opt
       
    
        
