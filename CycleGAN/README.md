# Cycle-Consistent Adversarial Networks (CycleGAN)

Tensorflow implementation of CycleGAN following this [paper](https://arxiv.org/abs/1703.10593). Referenced torch code can be found [here](https://github.com/junyanz/CycleGAN).

<img src="https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/CycleGAN_working.png" width="600" alt="CycleGAN model">

## Prerequisites

* Python 3.3+
* Tensorflow 
* pillow (PIL)
* matplotlib 
* (Optional) [Monet-Photo Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)
* (Optional) [Summer-Winter Yosemite Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip)

## Results 
Trained CycleGAN model on Monet-Photo Database and Summer-Winter Yosemite Database. 

### Photo to Monet Paintings

| Input Image | Output Image | Input Image | Output Image |
|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/orgB12.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/monetB12.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/orgB16.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/monetB16.png) 


### Monet Paintings to Photo 

| Input Image | Output Image | Input Image | Output Image |
|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/orgA16.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/realA16.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/orgA55.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/monet_100_200/realA55.png) 

### Summer to Winter Yosemite 

| Input Image | Output Image | Input Image | Output Image |
|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/sumA21.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/wintA21.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/sumA26.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/wintA26.png) 

### Winter to Summer Yosemite 

| Input Image | Output Image | Input Image | Output Image |
|:-----------:|:------------:|:-----------:|:------------:|
![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/wintB16.png)  |  ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/sumB16.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/wintB17.png) | ![](https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/sw_250_200/sumB17.png) 


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/LICENSE) file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
