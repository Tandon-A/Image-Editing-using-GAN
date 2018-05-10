# Deep Convolutional Generative Adversarial Network (DCGAN)

Tensorflow implementation of DCGAN following this [paper](https://arxiv.org/abs/1511.06434). Referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

<img src="https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/DCGAN/assets/DCGAN.png" width="800" alt="DCGAN Generator Architecture">

## Prerequisites

* Python 3.3+
* Tensorflow 1.6
* pillow (PIL)
* matplotlib 
* (Optional) [CelebA Face Database](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): Large Scale Face database used for training the model 


## Usage 

To train the model: 
```
 > python train_model.py --data_path celeba --input_fname_pattern .jpg --model_dir dcgan_model --sampled_images_dir gen_images_train
```
* data_path: Directory of the database folder having training images
* input_fname_pattern: Glob pattern of training images
* model_dir: Directory path to save checkpoints
* sampled_images_dir: Directory where images sampled from the generator (while training the model) are stored  


To test a trained model: 
```
> python sampling.py --model_dir dcgan_model\try_6 --sampled_dir gen_images_test
```
* model_dir: Model weights directory
* sampled_dir: Directory where images sampled from the generator (while testing the model) are stored


## Results 
Sampling through epochs during training. 

![Run 1](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/DCGAN/assets/run1_gif.gif "Generated Samples from epoch 0 to epoch 8 - Run1")

![Run 2](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/DCGAN/assets/run2_gif.gif "Generated Samples from epoch 0 to epoch 8 - Run2")

### Results after 6th epoch 

<img src="https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/DCGAN/assets/6_run.png" width="500" alt="Generated Samples after 6th Epoch">


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/LICENSE) file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
