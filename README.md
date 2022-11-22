# FaceMaskRecognize
A project for researching and createing a convolution neural network model that can recognize masked-face image.

# How to run this project

## Step 1: Download project
Clone project from github: https://github.com/lam1101999/FaceMaskRecognize  
Or Get project file from other sources

## Step 2: Install
Install Python: 
```
https://www.python.org
```
Install Library for project:
```
Install Cmake: https://cmake.org/download (requirement for dlib)
Download dlib: https://github.com/davisking/dlib
Change directory to dlib folder then typing: "python setup.py install" if there is any errors try "python setup.py install --no DLIB_GIF_SUPPORT"
Change directory to this project then typing: "pip install -r requirements.txt"
```
## Function 1: Clean and Wear mask
Note: This function will read image and extract face from images then wear mask on the face images.  
Image folder must follow this structure: one main folder, there will be several sub-folder inside main folder, each sub-folder will storages image of the same person.  
To run this function, in folder wear_mask, edit file FacialMaskDataset.py
![](https://github.com/lam1101999/FaceMaskRecognize/blob/master/image_github/unlock_clean_wear_mask.jpg)
![](https://github.com/lam1101999/FaceMaskRecognize/blob/master/image_github/Input_and_output_directory_clean.jpg)
![](https://github.com/lam1101999/FaceMaskRecognize/blob/master/image_github/Input_and_output_wear_mask.jpg)
Run file FacialMaskDataset.py:
```
python FacialMaskDatsetpy
```

## Function 2: Train model
Create and train a model from scratch. You can use "Clean and Wear mask function" to create your own dataset. My dataset will not be uploaded here because of github memory limitation. Dataset: Casia-Webface, AFD (for Asian people)  
Run file Train.ipynb  
This file is design to run on my personal GoogleColab so you must change some link and address for your account.

## Function 3: Face Recognize in real time
We uploaded our trained model in models/epoch49.h5 (in this project I called it Facenet)  
Newer version is in models/epoch101.h5 this model is trained with ArcFace and have better performance ( in this project I called it ArcFace)  
To use these models I create a supported class in product/MyModel. The constructor receive string to init model ("Facenet", "NewFacenet", "ArcFace")  
This class support some function like convert image into vector,.... We dont need to preprocess image and rewrite code.  
Make sure to edit the address of model in this class.  

We also create a GUI application to give an example how to use this model.  
Your computer must have a camera.  
```
python face_recognize.py
```
