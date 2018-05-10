# Cycle-Consistent Adversarial Networks (CycleGAN)

Tensorflow implementation of CycleGAN following this [paper](https://arxiv.org/abs/1703.10593). Referenced torch code can be found [here](https://github.com/junyanz/CycleGAN).

<img src="https://raw.githubusercontent.com/Tandon-A/Image-Editing-using-GAN/master/CycleGAN/assets/CycleGAN_working.png" width="600" alt="CycleGAN model">

## Prerequisites

* Python 3.3+
* Tensorflow 1.6
* pillow (PIL)
* matplotlib 
* (Optional) [Monet-Photo Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip)
* (Optional) [Summer-Winter Yosemite Database](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip)

## Usage

To train the model:
```
> python train_cycleGAN.py --data_path monet2photo --input_fname_pattern .jpg --model_dir cycleGAN_model --sampled_images_dir train_image_dir
```
* data_path: Path to parent directory of trainA and trainB folder
* input_fname_pattern: Glob pattern of training images
* model_dir: Directory name to save checkpoints
* sampled_images_dir: Directory where images sampled from the generator (while training the model) are stored  


To test the model:
```
> python test_cycleGAN.py --testA_image A01.jpg --testB_image B01.jpg --model_dir cycleGAN_model --sampled_imaes_dir test_image_dir
```
* testA_image: TestA Image Path
* testB_image: TestB Image Path 
* model_dir: Path to checkpoint folder
* sampled_images_dir: Directory where images sampled from the generator (while testing the model) are stored



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
