import numpy as np 
import tensorflow as tf 
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt 

#import the DeepConv Encoder model definition from model definition python file.
from encoder_model import Encoder


"""
Function to plot an image passed as an numpy array.
"""
def plot_it(img,title):
    if np.array_equal(img.max(),img.min()) == False:
        img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
    else:
        img = ((img - img.min())*255).astype(np.uint8)
    plt.imshow(img)
    plt.title(title)
    plt.show()

    
"""
Function to get image from path and rescale it to [-1,1]
"""
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    image = image.crop([30,40,168,178])
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)
    return image


"""
Function to get attribute encodings saved in a .txt file. Formed after running average.py. 
"""
def get_attr_encoding(file_path_attr,file_path_encoding):
    with open(file_path_encoding,"r") as f:
        enc = f.readlines()
    with open(file_path_attr,"r") as f:
        label = f.readlines()
    
    encoding = []
    for i in range(len(enc)):
        embed = enc[i].split(' ')
        encoding.append(np.array(embed[0:100]))
    encoding = np.array(encoding).astype(np.float32)
    attr = np.array(label[1].split(' ')[0:40])
    return encoding,attr


def test(encoder,encoder_dir,encoding,attr,data_files):
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())  
      saver.restore(sess,encoder_dir+"\\")
      img = np.array(
           [get_image_new(sample_file,64,64) for sample_file in data_files[0:64]]).astype(np.float32) 
      gen_en,log_en = sess.run([encoder.en_gen_img,encoder.en_logits],feed_dict={encoder.input_images:img})
      
      
      #Take encoding of first image 
      log = log_en[0]
      
      plot_it(img[0])
      plot_it(gen_en[0])
      
      #Manipulate the image encoding with attribute encodings to obtain different images. 
      for i in range(len(attr)):
          z_new = log + encoding[i]
          gen_img,_ = sess.run(encoder.generator(z_new,3,False))
          gen_img = np.reshape(gen_img,(64,64,3))
          plot_it(gen_img,attr[i])
     

def main(_): 
    
    if not os.path.exists(FLAGS.data_path):
        print ("Training Path doesn't exist")
    else:
        if not os.path.exists(FLAGS.encoder_dir):
            print ("Encoder model is not available at the specified path")
        else:
            if not os.path.exists(FLAGS.attribute_file):
                print ("Attribute Label File is not available at the specified path")
            else:
                if not os.path.exists(FLAGS.attr_encoding_file):
                    print ("Attribute Encoding file is not available at the specified path")
                else:     
                    #CelebA Face Database is used in this project. 
                    data_files = glob(str(FLAGS.data_path) +"\\"+str(FLAGS.input_fname_pattern))
                    encoding,attr = get_attr_encoding(FLAGS.attribute_file,FLAGS.attr_encoding_file)
                    batch_size = 64
                    z_dim = 100
                    lr_rate = 0.0002
                    beta1 = 0.5
                    alpha = 0.2
                    shape = 64,64,3
                    tf.reset_default_graph()
                    en_net = Encoder(lr_rate,shape,z_dim,batch_size,beta1,alpha)
                    test(en_net,FLAGS.encoder_dir,encoding,attr,data_files)
        


flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Directory of the database folder having training images")
flags.DEFINE_string("input_fname_pattern","*.jpg","Glob pattern of training images")
flags.DEFINE_string("encoder_dir",None,"Encoder directory to save checkpoints")
flags.DEFINE_string("attribute_file",None,"Path to attribute label .txt file")
flags.DEFINE_string("attr_encoding_file",None,"Path to attribute encoding .txt file. Formed after running avergae.py")
FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
