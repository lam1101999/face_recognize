import numpy as np
from tool.FileFunction import FileFunction
import os
import tensorflow as tf
from tool.FormatFunction import FormatFunction
from tool.GlobalValue import GlobalValue
import tensorflow_addons as tfa
import pickle
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import time
class Classify:
    def __init__(self, model, format_function):
        self.model = model
        self.file_function = FileFunction()
        self.format_function = format_function
    def init_model(self, model):
        self.model = model

    def embedding_data_one_label(self, dataset):
        encodes = self.model.predict(dataset)
        if encodes.any():
            encodes = np.average(encodes, axis=0 )
        return encodes
    def embedding_data_one_labelV2(self, data):
        encodes = self.model.predict(data)
        if encodes.any():
            encodes = np.average(encodes, axis=0 )
        return encodes

    def embedding_one_data_by_directory(self, one_data_directory, encoding_dictionary = dict(), probability_train = None):
        label =  one_data_directory.split(os.path.sep)[-1]
        path_each_file = self.file_function.getPath(one_data_directory)
        if(probability_train is not None):
            number_file = len(path_each_file)
            min = np.maximum(int(probability_train*number_file), 1) 
            path_each_file = path_each_file[:min]
        dataset_of_one_person = tf.data.Dataset.from_tensor_slices(path_each_file)
        dataset_of_one_person = dataset_of_one_person.map(self.format_function.get_label_as_string, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_of_one_person = dataset_of_one_person.map(self.format_function.process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_of_one_person = dataset_of_one_person.batch(10)
        embedding_one_person = self.embedding_data_one_label(dataset_of_one_person)
        encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary        

    def embedding_all_data_by_directory(self, all_data_directory, probability_train = None):
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)
        for each_folder in tqdm(path_to_each_folder, ascii=" *"):
            label  = each_folder.split(os.path.sep)[-1]
            path_each_file = self.file_function.getPath(each_folder)
            if(probability_train is not None):
                number_file = len(path_each_file)
                min = np.maximum(int(probability_train*number_file), 1) 
                path_each_file = path_each_file[:min]
            if len(path_each_file) <= 0:
                continue
            dataset_of_one_person = tf.data.Dataset.from_tensor_slices(path_each_file)
            dataset_of_one_person = dataset_of_one_person.map(self.format_function.get_label_as_string, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_of_one_person = dataset_of_one_person.map(self.format_function.process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_of_one_person = dataset_of_one_person.batch(10)
            embedding_one_person = self.embedding_data_one_label(dataset_of_one_person)
            encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary
    
    def embedding_all_data_by_directoryV2(self, all_data_directory, probability_train = None):
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)

        #work with iamge each person
        for each_folder in tqdm(path_to_each_folder, ascii=" *"):
            data = list()
            label  = each_folder.split(os.path.sep)[-1]
            # get all path image of this person
            path_each_files = self.file_function.getPath(each_folder)
            if(probability_train is not None):
                number_file = len(path_each_files)
                min = np.maximum(int(probability_train*number_file), 1) 
                path_each_files = path_each_files[:min]
            if len(path_each_files) <= 0:
                continue
            
            # Convert path image to image
            for path_each_file in path_each_files:
                try:
                    image  = self.format_function.open_and_process_image_Pillow(path_each_file)
                    data.append(image)
                except Exception:
                    continue

            data = np.asarray(data)   
            # get embedding
            embedding_one_person = self.embedding_data_one_labelV2(data)
            encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary
        
    def save_embedding_to_file(self, embedding, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(embedding, file)

    def load_embedding_from_file(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                encoding_dict = pickle.load(file)
        except FileNotFoundError:
            return dict()
        return encoding_dict

    def detect_one_image(self, image, embedding, thresh_hold = 4):
        image = np.expand_dims(image,0)
        encode = self.model.predict(image)[0]
        name = "unknow"
        distance = float("inf")
        for db_name, db_encode in embedding.items():
            dist = euclidean(db_encode, encode)
            if dist < thresh_hold and dist < distance:
                name = db_name
                distance = dist
        return name
    def detect_on_dataset(self, dataset, embedding, thresh_hold = 4):
        list_name = list()
        encode = self.model.predict(dataset)
        for each_encode in tqdm(encode, ascii=" *"):
            name = "unknow"
            distance = float("inf")
            for db_name, db_encode in embedding.items():
                dist = euclidean(db_encode, each_encode)
                if dist < thresh_hold and dist < distance:
                    name = db_name
                    distance = dist
            list_name.append(name)

        return list_name
    
    def evaluate(self, dataset, embedding, thresh_hold = 4):
        right_predict = 0
        list_real_label = list()
        for _, label_batch in tqdm(dataset, ascii=" *"):
            for label in label_batch:
                list_real_label.append(label)
        
        list_predict_label = self.detect_on_dataset(dataset, embedding, thresh_hold)

        for i in range(len(list_real_label)):
            if (list_predict_label[i] != "unknow" and list_real_label[i] == list_predict_label[i]):
                right_predict += 1
        return right_predict/len(list_real_label)

    

def main():
    global_value = GlobalValue(image_size=[120,120], batch_size = 512, shuffle_size = 1000, ratio_train = 0.8, ratio_test = 0.1, ratio_valid = 0.1, epochs = 40, small_epochs = 50,
                           image_each_class = 10)
    format_function = FormatFunction(global_value)
    load_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.getcwd()),"save_model","align_image_origin14"), custom_objects={"Addons>TripletSemiHardLoss":tfa.losses.TripletSemiHardLoss})
    classify = Classify(load_model,format_function)
    embed_10_person = classify.embedding_all_data_by_directory(os.path.join(os.path.dirname(os.getcwd()),"10_person"))
    print(len(embed_10_person))
if __name__ == "__main__":
    main()