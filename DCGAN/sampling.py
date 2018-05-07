import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import os
import scipy.misc

#import the DCGAN model definition from model definition python file. 
from dcgan_model import DCGAN


def sample(net,z_dim,model_dir,image_dir):
        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())  
          example_z = np.random.uniform(-1,1,size=[1,z_dim])    
          saver = tf.train.Saver()
          saver.restore(sess,model_dir+"\\")
          print ("generating image")
          img = sess.run(net.output_gen,feed_dict={net.input_z:example_z})
          img = np.reshape(img,(64,64,3))
          if np.array_equal(img.max(),img.min()) == False:
                 img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
          else:
                 img = ((img - img.min())*255).astype(np.uint8)
          scipy.misc.toimage(img, cmin=0.0, cmax=...).save(image_dir+"\\gen_img.jpg")
          plt.imshow(img)
          plt.show()
      
        
def main(_): 

    if not os.path.exists(FLAGS.model_dir):
        print ("Model Weights do not exist")
    else:          
        if not os.path.exists(FLAGS.sampled_dir):
            os.makedirs(FLAGS.sampled_dir)
                  
        z_dim = 100
        lr_rate = 0.0002
        beta1 = 0.5
        alpha = 0.2
        shape = 64,64,3
        tf.reset_default_graph()
        net = DCGAN(shape,z_dim,lr_rate,beta1,alpha)
        sample(net,z_dim,FLAGS.model_dir,FLAGS.sampled_dir)



flags = tf.app.flags
flags.DEFINE_string("model_dir","dcgan_model","Model weights directory")
flags.DEFINE_string("sampled_dir","sampled_images","Directory where images sampled from the generator (while testing the model) are stored")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()

