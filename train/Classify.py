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
        name = "unknown"
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
            name = "unknown"
            distance = float("inf")
            for db_name, db_encode in embedding.items():
                dist = cosine(db_encode, each_encode)
                if dist < thresh_hold and dist < distance:
                    name = db_name
                    distance = dist
            list_name.append(name)

        return list_name
    
    def evaluate(self, dataset, embedding, thresh_hold = 4):
        unknow_answer = 0
        right_answer = 0
        mis_answer = 0 
        list_real_label = list()
        for _, label_batch in tqdm(dataset, ascii=" *"):
            for label in label_batch:
                list_real_label.append(label)
        total_answer = len(list_real_label)
        list_predict_label = self.detect_on_dataset(dataset, embedding, thresh_hold)

        for i in range(len(list_real_label)):
            if (list_predict_label[i] == "unknown"):
                unknow_answer +=1
            elif (list_predict_label[i] == list_real_label[i]):
                right_answer +=1
            else:
                mis_answer +=1

        return right_answer, unknow_answer, mis_answer, total_answer
    
    def evaluate_using_confusion_matrix(self,datset,embedding, thresh_hold = 4):
        matrix = self.calculate_confusion_matrix(datset,embedding, thresh_hold)
        precision, recall, accuracy,f1 = self.calculate_precision_recall_accuracy_f1_from_matrix(matrix)
        return precision, recall, accuracy,f1
    
    def calculate_confusion_matrix(self, dataset, embedding, thresh_hold = 4):

        # Initialize Confusion Matrix (row is real label, column is predict label so precision is column, recall is row)
        total_label = len(embedding)
        matrix = np.zeros((total_label, total_label))


        # give label an offset to work with matrix
        off_set = 0
        off_set_dictionary = dict()
        for key in embedding.keys():
            off_set_dictionary[key] = off_set
            off_set+=1
        # Get list real label
        list_real_label = list()
        for _, label_batch in tqdm(dataset, ascii=" *"):
            for label in label_batch:
                list_real_label.append(label)
        # Get list predict label 
        list_predict_label = self.detect_on_dataset(dataset, embedding, thresh_hold)

        # Assign value to matrix
        for i in range(len(list_real_label)):
            real_label = list_real_label[i].numpy()
            real_label = real_label.decode("ascii")
            predict_label = list_predict_label[i]
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
    global_value = GlobalValue(image_size=[110,110], batch_size = 512, shuffle_size = 1000, ratio_train = 0.8, ratio_test = 0.1, ratio_valid = 0.1, epochs = 40, small_epochs = 50,
                           image_each_class = 10)
    format_function = FormatFunction(global_value)

    #call model by weight because computer cannot open model directly
    model_path = os.path.join(os.path.dirname(os.getcwd()),"models", "model49.h5")
    input_size = [global_value.IMAGE_SIZE[0], global_value.IMAGE_SIZE[1], 3]
    reload_model  = call_instance_FaceNet_with_last_isDense(input_size,10575)
    reload_model.load_weights(model_path)
    embedding_model = convert_train_model_to_embedding(reload_model)
    print("Done init model")
    classify = Classify(embedding_model,format_function)


    matrix = np.array([[1,0,0],
                        [0,5,0],
                        [0,0,9]])
    precision, recall, accuracy,f1 = classify.calculate_precision_recall_accuracy_f1_from_matrix(matrix)
    print(precision, recall, accuracy,f1)
if __name__ == "__main__":
    main()