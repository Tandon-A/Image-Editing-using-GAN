import tensorflow as tf 
#import the DCGAN model definition from model definition python file. 
from dcgan_model import DCGAN
import numpy as np
import matplotlib.pyplot as plt 


#Change model_dir to location where you want to save model weights 
model_dir = "D:\\Abhishek_Tandon\\sop_dl\\model\\dcgan\\tf\\model\\"
z_dim = 100
lr_rate = 0.0002
beta1 = 0.5
alpha = 0.2
max_iter = 15
shape = 64,64,3

with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      net = DCGAN(shape,z_dim,lr_rate,beta1,alpha)
      saver = tf.train.Saver()
	  
	  #change second argument of restore function to the path where weights and .meta file is stored
      saver.restore(sess,model_dir+"try_2_0_upd\\")
	  
	  #creating an embedding of size 1x100 to sample the generator 
      example_z = np.random.uniform(-1,1,size=[1,z_dim])    
      img = sess.run(net.output_gen,feed_dict={net.input_z:example_z})
      img = np.reshape(img,(64,64,3))
      if np.array_equal(img.max(),img.min()) == False:
          img = (((img - img.min())*255)/(img.max()-img.min())).astype(np.uint8)
      else:
          img = ((img - img.min())*255).astype(np.uint8)
      plt.imshow(img)
      plt.show()
	  
	  #to print all the variables in the graph 
      """
      for v in tf.all_variables():
          print (v.op.name + " ")
      """

	  
