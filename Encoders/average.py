import tensorflow as tf
import numpy as np 
from PIL import Image
from glob import glob 
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
Function to get the labels for the images in the database using the attribute label file   
"""
def avg_z(data_files,en_net,batch_size,encoder_dir):
    z_vect = []
    max_bs_len = int(len(data_files)/batch_size)*batch_size
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())  
      saver.restore(sess,encoder_dir+"\\")
      bs = 0
      while bs < max_bs_len:
          img = np.array(
                  [get_image_new(sample_file,64,64) for sample_file in data_files[bs:(bs+64)]]).astype(np.float32) 
          log_en = sess.run(en_net.en_logits,feed_dict={en_net.input_images:img})
          for log in log_en:
              z_vect.append(log)
          print ("appending %r" %(bs))
          bs = bs + 64          
    return z_vect


def get_labels(file_path):
    with open(file_path,"r") as f:
        label = f.readlines()
    i = 2
    labels = []
    while i< len(label):
        lab = []
        for k in label[i].split(' '):
            if k != '':
                lab.append(k)
        labels.append(lab)
        i = i+1
        if i % 100 == 0:
            print ("labelling =%r" %(i))
        
    labels = np.array(labels)
    attr = np.array(label[1].split(' '))
    return labels, attr


def create_avg_file(attr_label_file,data_files,encoder_dir,attr_encoding_file):
    batch_size = 64
    z_dim = 100
    lr_rate = 0.0002
    beta1 = 0.5
    alpha = 0.2
    shape = 64,64,3
    tf.reset_default_graph()
    
    en_net = Encoder(lr_rate,shape,z_dim,batch_size,beta1,alpha)
    
    z_vect = avg_z(data_files,en_net,batch_size,encoder_dir)
    print (len(z_vect))
    
    labels,attributes = get_labels(attr_label_file)
    
    for i in range(len(attributes)-1):
        has = []
        has_not = []
        for k in range(len(z_vect)):
            if int(labels[k][i+1])==1:
                has.append(z_vect[k])
            elif int(labels[k][i+1])==-1:
                has_not.append(z_vect[k])
        mean_has = 0
        mean_has_not = 0
        for j in range(len(has)):
            mean_has = mean_has + has[j]
        mean_has = mean_has/len(has)
        for j in range(len(has_not)):
            mean_has_not = mean_has_not + has_not[j]
        mean_has_not = mean_has_not/len(has_not)
        
        attr_i = mean_has - mean_has_not
        print ("attr encoded")
        attr_i = np.reshape(attr_i,(100))
        with open(attr_encoding_file,"a") as f:
            for i in range(attr_i.shape[0]):
                f.write(str(attr_i[i]) + " ")
            f.write("\n")



def main(_): 
    
    if not os.path.exists(FLAGS.data_path):
        print ("Dataste is not available at specified path")
    else:
        if not os.path.exists(FLAGS.encoder_dir):
            print ("Encoder model is not available at the specified path")
        else:
            if not os.path.exists(FLAGS.attribute_file):
                print ("Attribute Label file is not available at the specified path")
            else:
                #CelebA Face Database is used in this project. 
                data_files = glob(str(FLAGS.data_path) +"\\"+str(FLAGS.input_fname_pattern))
                create_avg_file(FLAGS.attribute_file,data_files,FLAGS.encoder_dir,FLAGS.attr_encoding_file)


flags = tf.app.flags
flags.DEFINE_string("data_path",None,"Directory of the database folder having training images")
flags.DEFINE_string("input_fname_pattern","*.jpg","Glob pattern of training images")
flags.DEFINE_string("encoder_dir",None,"Encoder directory to save checkpoints")
flags.DEFINE_string("attribute_file",None,"Path to attribute label .txt file")
flags.DEFINE_string("attr_encoding_file","attr_embed11.txt","Path to .txt file to store attribute encodings.")

FLAGS = flags.FLAGS
    
if __name__ == '__main__':
    tf.app.run()
