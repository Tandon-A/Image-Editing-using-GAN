import tensorflow as tf
import numpy as np 
from PIL import Image
from glob import glob 

#import the DeepConv Encoder model definition from model definition pyhton file.
from encoder_model import Encoder

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


#Function to calculate the encoding of the images present in the database
def avg_z(data_files,en_net,batch_size):
    z_vect = []
    max_bs_len = int(len(data_files)/batch_size)*batch_size
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer()) 
      #change second argument of restore function to the location where model weights are saved 
      saver.restore(sess,encoder_model_dir+"l2\\try_7_0\\")
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


#Function to get the labels for the images in the database using the attribute label file       
def get_labels(file_path):
    with open(file_path,"r") as f:
        label = f.readlines()
        
    #skip the first two lines 
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

#change encoder_model_dir to the parent directory of the encoder model weights 
encoder_model_dir = "encoders\\model\\"
#change attr_label_file to the path of attr_label_file 
attr_label_file = "dataset\\list_attr_celeba.txt"


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


en_net = Encoder(lr_rate,shape[1:],z_dim,batch_size,beta1,alpha)


z_vect = avg_z(data_files,en_net,batch_size)
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
    print (attr_i)
    #write the attribute encoding to a file for further use. Change path of output file. 
    with open(encoder_model_dir+"attr_embed_new.txt","a") as f:
        for i in range(attr_i.shape[0]):
            f.write(str(attr_i[i]) + " ")
        f.write("\n")





