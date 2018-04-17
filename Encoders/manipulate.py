import numpy as np 
import tensorflow as tf 
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt 

#import the DeepConv Encoder model definition from model definition pyhton file.
from encoder_model import Encoder

#Function to plot an image passed as an numpy array.
def plot_it(img):
    if np.array_equal(img.max(),img.min()) == False:
        img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
    else:
        img = ((img - img.min())*255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    

#Function to get image from path and rescale it to [-1,1]
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    image = image.crop([30,40,168,178])
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    image = np.multiply(image,2)
    return np.array(image)

#Function to get attribute encodings saved in a .txt file. Formed after running average.py. 
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


def test(encoder,encoding,attr,data_files):
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())  
      saver.restore(sess,encoder_model_dir+"l2\\try_7_0\\")
      img = np.array(
           [get_image_new(sample_file,64,64) for sample_file in data_files[0:64]]).astype(np.float32) 
      gen_en,log_en = sess.run([encoder.en_gen_img,encoder.en_logits],feed_dict={encoder.input_images:img})
      
      
      #Take encoding of first image 
      log = log_en[0]
      
      plot_it(img[0])
      plot_it(gen_en[0])
      
      #Manipulate the image encoding with attribute encodings to obtain different images. 
      for i in range(len(attr)):
          print (attr[i])
          z_new = log + encoding[i]
          gen_img,_ = sess.run(encoder.generator(z_new,3,False))
          gen_img = np.reshape(gen_img,(64,64,3))
          plot_it(gen_img)
     
    
    
#change attr_label_file to the path of attr_label_file     
attr_label_file = "dataset\\list_attr_celeba.txt"
#change encoder_model_dir to the parent directory of the encoder model weights. 
encoder_model_dir = "encoders\\model\\"

#change the second argument of the 'get_attr_encoding' function to the path of the attribute encoding file. 
encoding,attr = get_attr_encoding(attr_label_file,encoder_model_dir+"attr_embed_new.txt")

batch_size = 64
z_dim = 100
lr_rate = 0.0002
beta1 = 0.5
alpha = 0.2
max_iter = 25
#change data_files to the location where database is stored 
data_files = glob("dataset\\img\\img_align_celeba\\*.jpg")
shape = len(data_files),64,64,3
tf.reset_default_graph()
en_net = Encoder(lr_rate,shape[1:],z_dim,64,beta1,alpha)
test(en_net,encoding,attr,data_files)
