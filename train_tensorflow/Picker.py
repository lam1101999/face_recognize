import os
import random
import tensorflow as tf
import cv2
class Picker:
    def __init__(self):
        pass
    def init_from_directory(self, directory):
        self.directory = directory

    def init_from_dataset(self, dataset):
        self.dataset = dataset
    
    def pick_randomly_from_directory(self, label, mode_label_is_tensor = True):

        if mode_label_is_tensor:
            number_label = int(label.numpy())
            label = ("%7s"%number_label).replace(" ", "0")

        path_label = os.path.join(self.directory, label)
        list_file_in_label = [f for f in os.listdir(path_label) if os.path.isfile(os.path.join(path_label, f))]
        random_offset = random.randrange(0,len(list_file_in_label))
        random_path = os.path.join(path_label, list_file_in_label[random_offset])
        # convert path to image and processing image
        image = cv2.imread(random_path)
        image = cv2.resize(image,[190,190])
        image = image/255

        return tf.convert_to_tensor(image), tf.strings.to_number(tf.convert_to_tensor(label))

    def pick_randomly_from_dataset(self, param_label):
        result = None
        for data in self.dataset.filter(lambda img, label: tf.math.equal(label,param_label)).take(1):
            result = data
        return result


if __name__ == "__main__":
    current = os.path.dirname(os.getcwd())
    directory = os.path.join(current, "10_person")
    picker = Picker()
    picker.init_from_directory(directory)

    result = picker.pick_randomly_from_directory(tf.convert_to_tensor(45))
    print(result)