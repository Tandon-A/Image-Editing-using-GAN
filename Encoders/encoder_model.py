import tensorflow as tf 

"""
Class Definition of DeepConv Encoder as presented in paper "Photoshop 2.0:Generative Adversarial Networks for Photo Editing". 
"""


class Encoder:
    
    def __init__(self,lr_rate_en,input_shape,z_dim,no_of_im,beta1=0.5,alpha=0.2):
        self.lr_rate_en = tf.Variable(lr_rate_en,trainable=False,name="Encoder_lr")
        self.input_images = tf.placeholder(dtype=tf.float32,shape=(None,input_shape[0],input_shape[1],input_shape[2]),name="input_encoder")
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.im_num = no_of_im
        self.en_loss,self.en_gen_img,self.en_logits = self.model_loss(self.input_images,self.z_dim,self.input_shape[2],self.im_num)
        self.en_opt = self.model_opti(self.en_loss,self.lr_rate_en,beta1)
        
    """
    Encoder
    """        
    def encoder(self,images,alpha=0.2,reuse=False):
        w_init = tf.random_normal_initializer(mean=0.0,stddev = 0.02)
        with tf.variable_scope("encoder",reuse=reuse):
            
           conv1 = tf.layers.conv2d(images,64,5,2,padding='SAME',kernel_initializer=w_init)
           lrelu1 = tf.nn.leaky_relu(conv1,alpha=alpha)
           
           conv2 = tf.layers.conv2d(lrelu1,128,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm1 = tf.layers.batch_normalization(conv2,training=True)
           lrelu2 = tf.nn.leaky_relu(bn_norm1,alpha=alpha)
           
           conv3 = tf.layers.conv2d(lrelu2,256,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm2 = tf.layers.batch_normalization(conv3,training=True)
           lrelu3 = tf.nn.leaky_relu(bn_norm2,alpha=alpha)
           
           conv4 = tf.layers.conv2d(lrelu3,512,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm3 = tf.layers.batch_normalization(conv4,training=True)
           lrelu4 = tf.nn.leaky_relu(bn_norm3,alpha=alpha)
           
           conv5 = tf.layers.conv2d(lrelu4,512,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm4 = tf.layers.batch_normalization(conv5,training=True)
           lrelu5 = tf.nn.leaky_relu(bn_norm4,alpha=alpha)
           
           conv6 = tf.layers.conv2d(lrelu5,512,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm5 = tf.layers.batch_normalization(conv6,training=True)
           lrelu6 = tf.nn.leaky_relu(bn_norm5,alpha=alpha)
           
           conv7 = tf.layers.conv2d(lrelu6,512,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm6 = tf.layers.batch_normalization(conv7,training=True)
           lrelu7 = tf.nn.leaky_relu(bn_norm6,alpha=alpha)
           
           conv8 = tf.layers.conv2d(lrelu7,512,5,2,padding='SAME',kernel_initializer=w_init)
           bn_norm7 = tf.layers.batch_normalization(conv8,training=True)
           lrelu8 = tf.nn.leaky_relu(bn_norm7,alpha=alpha)
           
           conv9 = tf.layers.conv2d(lrelu8,512,5,2,padding='SAME',kernel_initializer=w_init)
           
           flat = tf.reshape(conv9,(-1,512))
           
           logits = tf.layers.dense(flat,100,kernel_initializer=w_init)
           
           return logits
    
    """
    Generator. Same as the generator of DCGAN. 
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
    Function to calculate loss for the encoder. Calculates L2 Loss between generated images and input images.
    Returns the loss, generated image and the input image encoding. 
    """
    def model_loss(self,input_images,z_dim,out_channel_dim,im_num):
        en_logits = self.encoder(input_images)
        gen_images,_ = self.generator(en_logits,out_channel_dim)
        gen_images = tf.reshape(gen_images,shape=(im_num,64,64,3))            
        en_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.squared_difference(input_images,gen_images),axis=3),axis=2),axis=1))
        return en_loss,gen_images,en_logits
    
    
    """
    Train the encoder using Adam Optimizer.
    """
    def model_opti(self,en_loss,lr_rate,beta1):
        
        train_vars = tf.trainable_variables()
        en_vars = [var for var in train_vars if var.name.startswith('encoder')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            en_train_opt = tf.train.AdamOptimizer(lr_rate,beta1=beta1).minimize(en_loss,var_list = en_vars)
        
        return en_train_opt
    
    
   
    
       
           
