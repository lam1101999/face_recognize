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
from train.FaceNet import call_instance_FaceNet_with_last_isDense, convert_train_model_to_embedding
class Classify:
    def __init__(self, model, format_function):
        self.model = model
        self.file_function = FileFunction()
        self.format_function = format_function
    def init_model(self, model):
        self.model = model
    def embedding_all_data_by_directory_no_normalization(self, all_data_directory, probability_train = None):
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
            embedding_one_person = self.embedding_data_one_label_no_normalization(dataset_of_one_person)
            encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary

    def embedding_data_one_label_no_normalization(self, dataset):
        encodes = self.model.predict(dataset)
        if encodes.any():
            encodes = np.average(encodes, axis=0 )
        return encodes

    def embedding_data_one_label(self, dataset):
        encodes = self.model.predict(dataset)
        if encodes.any():
            encodes = np.average(encodes, axis=0 )
            encodes = encodes/np.linalg.norm(encodes)
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
        total_image = len(self.file_function.getPath(all_data_directory))
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)
        start = time.time()
        #work with image each person
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
            embedding_one_person = self.embedding_data_one_label_no_normalization(data)
            encoding_dictionary[label] = embedding_one_person
        end = time.time()
        print("total time {} for {} images, average {}s per image".format(end - start, total_image, (end-start)/total_image))
        return encoding_dictionary
    
    def embedding_all_data_by_directoryV2(self, all_data_directory, probability_train = None):
        total_image = len(self.file_function.getPath(all_data_directory))
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)
        start = time.time()
        #work with image each person
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
            embedding_one_person = self.embedding_data_one_label(data)
            encoding_dictionary[label] = embedding_one_person
        end = time.time()
        print("total time {} for {} images, average {}s per image".format(end - start, total_image, (end-start)/total_image))
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
        start = time.time()
        image = np.expand_dims(image,0)
        encode = self.model.predict(image)[0]
        time_after_encoding = time.time()
        name = "unknown"
        distance = float("inf")
        for db_name, db_encode in embedding.items():
            dist = euclidean(db_encode, encode)
            if dist < thresh_hold and dist < distance:
                name = db_name
                distance = dist
        time_after_finding_label = time.time()
        return name, time_after_encoding - start, time_after_finding_label - time_after_encoding
    def detect_on_dataset(self, dataset, embedding, thresh_hold = 4):
        list_name = list()
        encode = self.model.predict(dataset)
        for each_encode in tqdm(encode, ascii=" *"):
            name = "unknown"
            distance = float("inf")
            for db_name, db_encode in embedding.items():
                dist = cosine(db_encode, each_encode)
                if dist < thresh_hold and dist < distance:
                    name = db_name
                    distance = dist
            list_name.append(name)

        return list_name
    
    def evaluate(self, list_path, embedding, thresh_hold = 4):

        total_encoding_duration = 0
        total_finding_duration = 0
        unknow_answer = 0
        right_answer = 0
        mis_answer = 0 
        total_answer = len(list_path)

        for each_path in tqdm(list_path, ascii=" *"):
            actual_label = each_path.split(os.path.sep)[-2]
            image = self.format_function.open_and_process_image_Pillow(each_path)
            predicted_label, encoding_duration, finding_duration = self.detect_one_image(image, embedding, thresh_hold)            
            if (predicted_label == "unknown"):
                unknow_answer +=1
            elif (predicted_label == actual_label):
                right_answer +=1
            else:
                mis_answer +=1

            total_encoding_duration += encoding_duration
            total_finding_duration += finding_duration
        print("database have {} vector, evaluate {} image, average_time_encoding {}, average_time_finding {}".format(len(embedding),total_answer, total_encoding_duration/total_answer, total_finding_duration/total_answer))
        return right_answer, unknow_answer, mis_answer, total_answer
    
    def evaluate_using_confusion_matrix(self,list_path,embedding, thresh_hold = 4):
        matrix = self.calculate_confusion_matrix(list_path,embedding, thresh_hold)
        precision, recall, accuracy,f1 = self.calculate_precision_recall_accuracy_f1_from_matrix(matrix)
        return precision, recall, accuracy,f1
    
    def calculate_confusion_matrix(self, list_path, embedding, thresh_hold = 4):

        # Initialize Confusion Matrix (row is real label, column is predict label so precision is column, recall is row)
        total_label = len(embedding)
        matrix = np.zeros((total_label, total_label))


        # give label an offset to work with matrix
        off_set = 0
        off_set_dictionary = dict()
        for key in embedding.keys():
            off_set_dictionary[key] = off_set
            off_set+=1
        print(len(list_path))
        # predict label
        for each_path in tqdm(list_path, ascii=" *"):
            real_label = each_path.split(os.path.sep)[-2]
            image = self.format_function.open_and_process_image_Pillow(each_path)
            predict_label, _, _ = self.detect_one_image(image, embedding, thresh_hold)
            # Assign value to matrix
            offset_row = off_set_dictionary[real_label]
            offset_column = off_set_dictionary[predict_label]
            matrix[offset_row, offset_column] = matrix[offset_row, offset_column]+1

        return matrix
    def calculate_precision_recall_accuracy_f1_from_matrix(self, confusion_matrix):
        one_size_matrix = len(confusion_matrix)

        #calculate precision we will calculate precision of each label then get average, ignore that label if that label does not have prediction(sum column = 0)
        total_label_have_precision = 0
        total_precision = 0
        for i in range(one_size_matrix):
            total_value_predict_as_i = np.sum(confusion_matrix[:,i])
            total_value_i_predict_as_i = confusion_matrix[i,i]
            if total_value_predict_as_i != 0:
                precision_i = total_value_i_predict_as_i/total_value_predict_as_i
                total_precision += precision_i
                total_label_have_precision += 1
        precision = total_precision/total_label_have_precision

        #calculate recall we will calculate recall of each label then get average, ignore that label if that label does not have recall(sum row = 0)
        total_label_have_recall = 0
        total_recall = 0
        for i in range(one_size_matrix):
            total_value_is_i = np.sum(confusion_matrix[i])
            total_value_i_predict_as_i = confusion_matrix[i,i]
            if total_value_is_i != 0:
                recall_i = total_value_i_predict_as_i/total_value_is_i
                total_recall += recall_i
                total_label_have_recall += 1
        recall = total_recall/total_label_have_recall

        #calculate accuracy
        total_right_answers = 0
        for i in range(one_size_matrix):
            total_right_answers += confusion_matrix[i,i]
        total_answers = np.sum(confusion_matrix)
        accuracy = total_right_answers/total_answers

        #calculate f1
        f1 = 2*(precision*recall)/(precision+recall)

        return precision, recall, accuracy,f1




    

def main():
    pass
if __name__ == "__main__":
    main()