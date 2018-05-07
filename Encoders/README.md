# Convolutional Encoder

Tensorflow implementation of DeepConv Encoder as presented in the paper - [Photoshop 2.0: Generative Adversarial Networks for Photo Editing](http://cs231n.stanford.edu/reports/2017/pdfs/305.pdf).


<img src="https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/deep_conv.PNG" width = "150" alt="DeepConv Encoder">


## Prerequisites

* Python 3.3+
* Tensorflow 
* pillow (PIL)
* matplotlib 
* (Optional) [CelebA Face Database](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): Large Scale Face database used for training the model 

## Usage 

To train the Encoder model:
```
> python train_encoder.py --data_path celeba --input_fname_ppatern .jpg --dcgan_model_dir dcgan_model --encoder_dir encoder_model
```
* data_path: Directory of the database folder having training images
* input_fname_pattern: Glob pattern of training images
* dcgan_model_dir: DCGAN model directory having model weights
* encoder_dir: Encoder directory to save checkpoints

To compute attribute encodings: 
```
> python average.py --data_path celeba --input_fname_ppatern .jpg --encoder_dir encoder_model --attribute_file attr_labels.txt --attr_encoding_file attr_embed.txt
```
* data_path: Directory of the database folder having training images
* input_fname_pattern: Glob pattern of training images
* encoder_dir: Encoder directory to save checkpoints
* attribute_file: Label file of the database
* attr_encoding_file: File to save attribute encodings  


To manipulate images:
```
> python manipulate.py --data_path celeba --input_fname_ppatern .jpg --encoder_dir encoder_model --attribute_file attr_labels.txt --attr_encoding_file attr_embed.txt
```

## Results 
Results of encoder model after 6th epoch. 

Orignal Image - Generated Image (after encoding the image)

![Org Img 1](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex1/data.png "Orignal Image 1")
![Gen Img 1](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex1/en_gen.png "Generated Image 1")

![Org Img 2](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex3/data.png "Orignal Image 2")
![Gen Img 2](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex3/en_gen.png "Generated Image 2")

![Org Img 3](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/data.png "Orignal Image 3")
![Gen Img 3](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/en_gen.png "Generated Image 3")


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/LICENSE) file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
