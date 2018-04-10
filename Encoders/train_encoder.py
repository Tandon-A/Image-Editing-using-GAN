import tensorflow as tf 
import numpy as np 
from glob import glob
import random
from PIL import Image
import matplotlib.pyplot as plt 
#import the DeepConv Encoder model definition from model definition pyhton file.
from encoder_model import Encoder


#Function to get image from path and rescale it to [-1,1]
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
    #crop image to reduce clutter
    image = image.crop([30,40,168,178])
    #Resizing image to smaller size -- 64 x 64 generally 
    image = image.resize([width,height],Image.BILINEAR)
    image = np.array(image,dtype=np.float32)
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
    #scaling image to [-1,1]
    image = np.multiply(image,2)    
    return np.array(image)

#Train Encoder. Uses trained Generator of DCGAN model. 
def train(en_net,max_iter,batch_size,data_files,lr_rate,beta1,shape,z_dim):
    saver = tf.train.Saver(max_to_keep=None)
    random.shuffle(data_files)
    max_bs_len = int(len(data_files)/batch_size)*batch_size
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        #Restore weights of generator from trained DCGAN model. 
        saver2 = tf.train.Saver(var_list=gen_vars)
        #change second argument of restore function to path where weights and .meta file of DCGAN model is stored 
        saver2.restore(sess,dcgan_model_dir+"try_6_3000\\")
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
                    
                if step % 100 == 0:
                   plt.imshow(batch_images[-1])
                   plt.show()
                   gen_images, logs = sess.run([en_net.en_gen_img,en_net.en_logits],feed_dict={en_net.input_images:batch_images})
                   plt.imshow(gen_images[-1])
                   plt.show()
                   
                   
                if step % 1500 == 0:
                    dir_path =  encoder_model_dir + "try_" + str(epoch)+"_"+str(step)+"\\"
                    saver.save(sess,dir_path,write_meta_graph=True)
                    with open(dir_path+"graph.pb",'wb') as f:
                        f.write(sess.graph_def.SerializeToString())
                    print ("### Model weights Saved epoch = %r , step = %r ###" %(epoch,step))
                
                                     
                else: 
                    print ("calc")
                bs = bs + batch_size
                step = step +1 
            epoch = epoch + 1
                    
#Change dcgan_model_dir to parent directory where weights of DCGAN model are saved.                     
dcgan_model_dir = "DCGAN\\model_weights\\"
#Change encoder_model_dir to directory where you want to save the encoder model weights. 
encoder_model_dir = "D:\\Abhishek_Tandon\\sop_dl\\model\\encoders\\model\\l2\\"

batch_size = 64
z_dim = 100
lr_rate = 0.0002
beta1 = 0.5
alpha = 0.2
max_iter = 25
#Change data_files to loaction where database is saved. CelebA Face Database is used in this project. 
data_files = glob("dataset\\img\\img_align_celeba\\*.jpg")
shape = len(data_files),64,64,3
tf.reset_default_graph()


en_net = Encoder(lr_rate,shape[1:],z_dim,batch_size,beta1,alpha)

train(en_net,max_iter,batch_size,data_files,lr_rate,beta1,shape,z_dim)
