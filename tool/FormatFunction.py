import tensorflow as tf
import os
import numpy as np
import cv2
from PIL import Image
class FormatFunction:

    def __init__(self, global_value) -> None:
        self.global_value = global_value
        self.rng = tf.random.Generator.from_non_deterministic_state()

    def get_label_as_string(self,file_path):
        return file_path, tf.strings.split(file_path, os.path.sep)[-2]

    def get_label_as_number(self,file_path):
        return file_path, tf.strings.to_number(tf.strings.split(file_path, os.path.sep)[-2])
    
    def process_image(self,file_path, label = None):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels = 3)
        image = tf.image.resize(image, self.global_value.IMAGE_SIZE)
        image = image/255

        return image,label
    
    def process_image_without_label(self,file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels = 3)
        image = tf.image.resize(image, self.global_value.IMAGE_SIZE)
        image = image/255

        return image
            
    
    def process_imagev2(self, image, label = "null"):
        """_summary_ this function work like process_image function but the input is image, not path of image

        Args:
            image (_type_): _description_
            label (str, optional): _description_. Defaults to "null".
        """
        image = cv2.resize(image, dsize=self.global_value.IMAGE_SIZE)
        image = image/255

        return image, label
    
    def open_and_process_image_Pillow(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize(self.global_value.IMAGE_SIZE)
        image = np.asarray(image)
        image = image/255
        return image

    def scale(self,image,label):
        return image/255,label

    def format_channel(self,image,label):
        if (image.shape[2] == 1):
            image = tf.concat([image,image, image],2)
        return image,label


    def get_dataset_partition(self, dataset, train_percentage, test_percentage, valid_percentage = 0, shuffle = False, shuffle_size = 100):
        dataset_size = len(dataset)

        if shuffle == True:
            dataset = dataset.shuffle(shuffle_size)
            
        train_size = int(dataset_size*train_percentage)
        test_size = int(dataset_size*test_percentage)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size).take(test_size)
        valid_dataset = dataset.skip(train_size).skip(test_size)
        return train_dataset, test_dataset, valid_dataset

    def shape_dataset(self,dataset):
        dataset_to_numpy =  list(dataset.as_numpy_iterator())
        shape = tf.shape(dataset_to_numpy)
        print(shape)

    def format_gray_image(self, image, label):
        image = tf.cast(image, tf.float32) / 255.
        image = tf.concat([image,image,image],2)
        image = tf.image.resize(image, self.global_value.IMAGE_SIZE)
        return image, label
    
    def split_image_label(self, image, label):
        return image, label
    
    def augment_data(self, image, label):
        seed = self.rng.make_seeds(2)[0]
        
        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
        image = tf.image.stateless_random_contrast(image, lower=0.2, upper=0.5, seed=seed)
        image = tf.image.stateless_random_crop(image, size = [int(self.global_value.IMAGE_SIZE[0] * (80/100)),int(self.global_value.IMAGE_SIZE[1]*(80/100)), 3], seed=seed)
        image = tf.image.resize(image, self.global_value.IMAGE_SIZE)
        
        image = tf.image.stateless_random_flip_left_right(image, seed = seed)
        image = tf.image.stateless_random_flip_up_down(image, seed = seed)
        image = tf.image.stateless_random_jpeg_quality(image, 75, 95, seed)

        return image, label
    
    def augment_data_without_label(self, image):
        seed = self.rng.make_seeds(2)[0]
        
        image = tf.image.stateless_random_brightness(image, max_delta=0.2, seed=seed)
        image = tf.image.stateless_random_contrast(image, lower=0.2, upper=0.5, seed=seed)
        image = tf.image.stateless_random_crop(image, size = [int(self.global_value.IMAGE_SIZE[0] * (80/100)),int(self.global_value.IMAGE_SIZE[1]*(80/100)), 3], seed=seed)
        image = tf.image.resize(image, self.global_value.IMAGE_SIZE)
        
        image = tf.image.stateless_random_flip_left_right(image, seed = seed)
        image = tf.image.stateless_random_flip_up_down(image, seed = seed)
        image = tf.image.stateless_random_jpeg_quality(image, 75, 95, seed)

        return image
    def get_label_dict(self,img_dir):
        label_dict = dict()
        count = 0
        for obj in os.scandir(img_dir):
            if obj.is_dir():
                label_dict[obj.name] = count
                count += 1
        if count == 0:
            print("No dir in the ",img_dir)
            return None
        else:
            return label_dict

    def get_label_index(self,label, label_dict):
        return label_dict[label]
