import tensorflow as tf 
import numpy as np 
from glob import glob
import random
from PIL import Image
import os  
import scipy.misc

#import the DCGAN model definition from model definition python file. 
from dcgan_model import DCGAN


"""
Function to get image from path and rescale them to [-1,1]
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    #crop image to reduce clutter 
    image = image.crop((30,40,168,178))
    #Resizing image to smaller size -- 64 x 64 generally  
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)	
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    #scaling image to [-1,1]
    image = np.multiply(image,2)
    return image

#Train DCGAN 
def train(net,max_iter,batch_size,data_files,model_dir,image_dir,lr_rate,beta1,shape,z_dim):
    saver = tf.train.Saver(max_to_keep=None)
    
    random.shuffle(data_files)
    max_bs_len = int(len(data_files)/batch_size)*batch_size
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess,model_dir+"try_5_3000\\")
        epoch = 0
        while epoch < (max_iter):
            bs = 0
            step = 0
            while bs < max_bs_len:
                batch_files = data_files[bs:(bs+batch_size)]
                batch_images = np.array(
           [get_image_new(sample_file,64,64) for sample_file in batch_files]).astype(np.float32) 
                
                batch_z = np.random.normal(loc=0.0,scale=1.0,size=(batch_size,z_dim))
                
                sess.run([net.disc_opt,net.gen_opt],feed_dict={net.input_real:batch_images,net.input_z:batch_z})
                
                if step % 20 == 0:
                    train_disc_loss = net.disc_loss.eval({net.input_real:batch_images,net.input_z:batch_z})
                    train_gen_loss = net.gen_loss.eval({net.input_real:batch_images,net.input_z:batch_z})
                    print ("Epoch = %r, step = %r, disc_loss = %r , gen_loss = %r" % (epoch,step,train_disc_loss,train_gen_loss))
                
                if step % 100 == 0:
                    example_z = np.random.uniform(-1,1,size=[1,z_dim])    
                    img = sess.run(net.output_gen,feed_dict={net.input_z:example_z})
                    img = np.reshape(img,(64,64,3))
                    if np.array_equal(img.max(),img.min()) == False:
                        img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
                    else:
                        img = ((img - img.min())*255).astype(np.uint8)
                    
                    scipy.misc.toimage(img, cmin=0.0, cmax=...).save(image_dir+"\\img_"+str(epoch)+"_"+str(step)+".jpg")
                    
                else: 
                    print ("step = %r" %(step))
                bs = bs + batch_size
                step = step +1 
                
            dir_path =  model_dir + "try_" + str(epoch)+"\\"
            saver.save(sess,dir_path,write_meta_graph=True)
            print ("### Model weights Saved epoch = %r  ###" %(epoch))
            epoch = epoch + 1
            
            
def main(_): 
      

    if not os.path.exists(FLAGS.data_path):
        print ("Training Path doesn't exist")
    else:
            
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        if not os.path.exists(FLAGS.sampled_images_dir):
            os.makedirs(FLAGS.sampled_images_dir)
    
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
                
        net = DCGAN(shape,z_dim,lr_rate,beta1,alpha)
        
        train(net,max_iter,batch_size,data_files,FLAGS.model_dir,FLAGS.sampled_images_dir,lr_rate,beta1,shape,z_dim)


flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Directory of the database folder having training images")
flags.DEFINE_string("input_fname_pattern","*.jpg","Glob pattern of training images")
flags.DEFINE_string("model_dir","dcgan_model","Directory path to save checkpoints")
flags.DEFINE_string("sampled_images_dir","sampled_images","Directory where images sampled from the generator (while training the model) are stored")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
