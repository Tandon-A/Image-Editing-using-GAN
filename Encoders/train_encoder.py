import tensorflow as tf 
import numpy as np 
from glob import glob
import random
from PIL import Image 
import os 

#import the DeepConv Encoder model definition from model definition python file.
from encoder_model import Encoder

"""
Function to get image from path and rescale it to [-1,1]
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    #crop image to reduce clutter -- specifically for CelebA Face Database. Might need to tune for other databases. 
    image = image.crop([30,40,168,178])
    #Resizing image to smaller size -- 64 x 64 generally 
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    #scaling image to [-1,1]
    image = np.multiply(image,2)    
    return image

"""
Train Encoder. Uses trained Generator of DCGAN model. 
"""
def train(en_net,max_iter,batch_size,data_files,dcgan_model_dir,encoder_dir,lr_rate,beta1,shape,z_dim):
    saver = tf.train.Saver(max_to_keep=None)
    random.shuffle(data_files)
    max_bs_len = int(len(data_files)/batch_size)*batch_size
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess,encoder_model_dir+"try_7_0\\")
        
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        saver2 = tf.train.Saver(var_list=gen_vars)
        saver2.restore(sess,dcgan_model_dir +"\\")
        
        epoch = 0
        while epoch < (max_iter):
            bs = 0
            step = 0
            while bs < max_bs_len:
                batch_files = data_files[bs:(bs+batch_size)]
                batch_images = np.array(
           [get_image_new(sample_file,64,64) for sample_file in batch_files]).astype(np.float32) 
                
                sess.run(en_net.en_opt,feed_dict={en_net.input_images:batch_images})
                             
                if step % 20 == 0:
                    train_en_loss= sess.run([en_net.en_loss],feed_dict={en_net.input_images:batch_images})
                    print ("Epoch = %r, step = %r, en_loss = %r " % (epoch,step,train_en_loss))
                else: 
                    print ("step = %r" %(step))
                    
                bs = bs + batch_size
                step = step +1 
                
            dir_path =  encoder_dir + "try_" + str(epoch)+"\\"
            saver.save(sess,dir_path,write_meta_graph=True)
            print ("### Model weights Saved epoch = %r  ###" %(epoch))
            epoch = epoch + 1


def main(_):   
    if not os.path.exists(FLAGS.data_path):
        print ("Training Path doesn't exist")
    else:
        if not os.path.exists(FLAGS.dcgan_model_dir):
            print ("DCGAN model is not available at the specified path")
        else:            
            if not os.path.exists(FLAGS.encoder_dir):
                os.makedirs(FLAGS.encoder_dir)
            batch_size = 64
            z_dim = 100
            lr_rate = 0.0002
            beta1 = 0.5
            alpha = 0.2
            max_iter = 25
            #CelebA Face Database is used in this project. 
            data_files = glob(str(FLAGS.data_path) +"\\"+str(FLAGS.input_fname_pattern))
            shape = 64,64,3
            tf.reset_default_graph()

            en_net = Encoder(lr_rate,shape,z_dim,batch_size,beta1,alpha)

            train(en_net,max_iter,batch_size,data_files,FLAGS.dcgan_model_dir,FLAGS.encoder_dir,lr_rate,beta1,shape,z_dim)


flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Directory of the database folder having training images")
flags.DEFINE_string("input_fname_pattern","*.jpg","Glob pattern of training images")
flags.DEFINE_string("dcgan_model_dir",None,"DCGAN model directory having model weights")
flags.DEFINE_string("encoder_dir","encoder_model","Encoder directory to save checkpoints")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
