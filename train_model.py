import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from glob import glob
import random
from PIL import Image 
#import the DCGAN model definition from model definition python file. 
from dcgan_model import DCGAN


#Function to get image from path and rescale them to [-1,1]
def get_image_new(image_path,width,height):
    image = Image.open(image_path)
	#Resizing image to smaller size -- 64 x 64 generally  
    image = image.resize([width,height],Image.BILINEAR)
	#crop image to reduce clutter 
	image = image.crop((30,40,168,178))
    image = np.array(image,dtype=np.float32)	
    image = np.divide(image,255)
    image = np.subtract(image,0.5)
	#scaling image to [-1,1]
    image = np.multiply(image,2)
    return np.array(image)

#Train DCGAN 
def train(net,max_iter,batch_size,data_files,lr_rate,beta1,shape,z_dim):
    saver = tf.train.Saver()
    random.shuffle(data_files)
    max_bs_len = int(len(data_files)/batch_size)*batch_size
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(max_iter):
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
                    plt.imshow(img)
                    plt.show()
                
                if step % 500 == 0:
                    saver.save(sess,model_dir + "try_" + str(epoch)+"_"+str(step) +"_upd\\")
                    print ("### Model weights Saved epoch = %r , step = %r ###" %(epoch,step))
                    
                    
                else: 
                    print ("calc")
                bs = bs + batch_size
                step = step +1 
                    
#Change model_dir to location where you want to save model weights                    
model_dir = "D:\\Abhishek_Tandon\\sop_dl\\model\\dcgan\\tf\\model\\"
batch_size = 64
z_dim = 100
lr_rate = 0.0002
beta1 = 0.5
alpha = 0.2
max_iter = 25

"""
Change data_files to loca where database is saved. CelebA Face Database is used in this project. 
"""
data_files = glob("D:\\Abhishek_Tandon\\sop_dl\\dataset\\img\\img_align_celeba\\*.jpg")
shape = len(data_files),64,64,3
tf.reset_default_graph()


net = DCGAN(shape[1:],z_dim,lr_rate,beta1,alpha)

train(net,max_iter,batch_size,data_files,lr_rate,beta1,shape,z_dim)
    
