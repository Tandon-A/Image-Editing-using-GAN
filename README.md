# Image-Editing-using-GAN
Developing application to edit images to have desired attributes using Generative Adversarial Networks (GANs). 

The objective is to have answers to questions such as - 
* How would this person look if he was wearing a hat? 
* How would this place look in winter season? 

![Img 1](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/assets/winter-summer.png "Objective Image")

## Models

### Deep Convolutional Generative Adversarial Network (DCGAN) 

Provided implementation of DCGAN model in tensorflow. [Read here](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/DCGAN/)

Trained an encoder model(DeepConv Encoder) on top of the DCGAN model to encode an image and then pass it to the geneartor to produce image which is same as that of the input image. [Read here](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/Encoders/)

Used the Encoder-Generator model to manipulate images to have desired attributes. 


| Orignal Image            |  Encoded Image  | Male     |  High Cheekbones | Blond Hair | Bangs    |
:-------------------------:|:---------------:|:--------:|:----------------:|:----------:|:--------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/data.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/en_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/male_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/high_cheekbones_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/blond_hair_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex5/bangs_gen.png)


| Orignal Image            |  Encoded Image  | Bald     |5 o' Clock Shadow |Double Chin |Pale Skin |
:-------------------------:|:---------------:|:--------:|:----------------:|:----------:|:--------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex2/data.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex2/en_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex2/bald_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex2/5_o_clock_shadow_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex2/double_chin_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex2/pale_skin_gen.png)


| Orignal Image            |  Encoded Image  | Smiling  |  Bushy Eyebrows  | Mustache   | Wavy Hair|
:-------------------------:|:---------------:|:--------:|:----------------:|:----------:|:--------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex4/data.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex4/en_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex4/smiling_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex4/bushy_eyebrows_gen.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex4/mustache_en.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/Encoders/assets/ex4/wavy_hair_gen.png)



### Cycle Consistent Adversarial Networks (CycleGAN)

Provided implementation of CycleGAN model in tensorflow. [Read here](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/CycleGAN/)

Trained the model on Monet-Photo Database and Summer-Winter Database. (with increased training images per epoch)

#### Monet Paintings to Photo 

| Input Image | Output Image | Input Image | Output Image | Input Image | Output Image | 
|:-----------:|:------------:|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/orgA2.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/realA2.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/orgA13.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/realA13.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/orgA21.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/realA21.png) 

#### Photo to Monet Paintings 

| Input Image | Output Image | Input Image | Output Image | Input Image | Output Image | 
|:-----------:|:------------:|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/orgB2.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/monetB2.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/orgB15.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/monetB15.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/orgB26.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_250_200/monetB26.png) 


## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
