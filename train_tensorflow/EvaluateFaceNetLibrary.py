from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from tool.FormatFunction import FormatFunction
from tool.GlobalValue import GlobalValue
from tool.FileFunction import FileFunction
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine, euclidean
import pickle



class ClassifyForPytorch:
    def __init__(self, model, format_function):
        self.model = model
        self.file_function = FileFunction()
        self.format_function = format_function
        self.transform = transforms.ToTensor()
    def init_model(self, model):
        self.model = model

    def embedding_data_one_label(self, dataset):
        encodes = self.model(dataset).detach()
        encodes = normalize(encodes)
        if encodes.any():
            encodes = np.average(encodes, axis=0 )
        return encodes

    def embedding_one_data_by_directory(self, one_data_directory, encoding_dictionary = dict(), probability_train = None):
        label =  one_data_directory.split(os.path.sep)[-1]
        path_each_files = self.file_function.getPath(one_data_directory)
        if(probability_train is not None and not path_each_file):
            number_file = len(path_each_file)
            min = np.maximum(int(probability_train*number_file), 1) 
            path_each_file = path_each_file[:min]
        list_image = []
        for path_each_file in path_each_file:
            try:
                image = self.format_function.open_and_process_image_Pillow(path_each_file)
                image = self.transform(image)
                image = image.float()
                list_image.append(image)
            except Exception:
                continue
        list_image_as_tensor = torch.stack(list_image,0)
        embedding_one_person = self.embedding_data_one_label(list_image_as_tensor)
        encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary        
    
    def embedding_all_data_by_directory(self, all_data_directory, probability_train = None):
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)

        #work with image each person
        for each_folder in tqdm(path_to_each_folder, ascii=" *"):
            list_image = list()
            label  = each_folder.split(os.path.sep)[-1]
            # get all path image of this person
            path_each_files = self.file_function.getPath(each_folder)
            if(probability_train is not None and not path_each_files):
                number_file = len(path_each_files)
                min = np.maximum(int(probability_train*number_file), 1) 
                path_each_files = path_each_files[:min]
            if len(path_each_files) <= 0:
                continue
            # Convert path image to image
            for path_each_file in path_each_files:
                try:
                    image  = self.format_function.open_and_process_image_Pillow(path_each_file)
                    image = self.transform(image)
                    image = image.float()
                    list_image.append(image)
                except Exception:
                    continue
            list_image_as_tensor = torch.stack(list_image,0)          
            # get embedding
            embedding_one_person = self.embedding_data_one_label(list_image_as_tensor)
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
        image = self.transform(image)
        image = image.float()
        encode = self.model(image.unsqueeze(0)).detach()
        # encode = normalize(encode)
        encode = encode[0]
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
        encode = self.model(dataset).detach()
        # encode = normalize(encode)
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
    
    def evaluate(self, list_path, embedding, thresh_hold = 4):
        unknow_answer = 0
        right_answer = 0
        mis_answer = 0
        total_answer = len(list_path)

        for each_path in tqdm(list_path, ascii=" *"):
            actual_label = each_path.split(os.path.sep)[-2]
            image = self.format_function.open_and_process_image_Pillow(each_path)
            predicted_label = self.detect_one_image(image, embedding, thresh_hold)
            if (predicted_label == "unknow"):
                unknow_answer +=1
            elif (predicted_label == actual_label):
                right_answer +=1
            else:
                mis_answer +=1
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

        # Predict label
        for each_path in tqdm(list_path, ascii=" *"):
            real_label = each_path.split(os.path.sep)[-2]
            image = self.format_function.open_and_process_image_Pillow(each_path)
            predict_label = self.detect_one_image(image, embedding, thresh_hold)

            #Assign value to matrix    
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

    #Init value
    global_value = GlobalValue(image_size=[160,160], batch_size = 512, shuffle_size = 1000, ratio_train = 0.8, ratio_test = 0.1, ratio_valid = 0.1, epochs = 40, small_epochs = 50,
                           image_each_class = 10)
    format_function = FormatFunction(global_value)
    file_function = FileFunction()
    model = InceptionResnetV1(pretrained='casia-webface').eval()
    classify = ClassifyForPytorch(model,format_function)
    data_directory1 = os.path.join(os.path.dirname(os.getcwd()), "dataset", "lfw_align")
    data_directory2 = os.path.join(os.path.dirname(os.getcwd()), "dataset", "lfw_mask")
    save_path  = os.path.join(os.path.dirname(os.getcwd()), "data_base_encoding", "database_embedding_facenet_pytorch.pkl")

    # get embedding
    # database_embedding_facenet_pytorch = classify.embedding_all_data_by_directory(data_directory1)
    # classify.save_embedding_to_file(database_embedding_facenet_pytorch, save_path)
    database_embedding_facenet_pytorch = classify.load_embedding_from_file(save_path)

    # get image path
    list_path = file_function.getPath(data_directory1)
    list_path_mask = file_function.getPath(data_directory2)
    # Evaluate
    precision, recall, accuracy,f1 = classify.evaluate_using_confusion_matrix(list_path, database_embedding_facenet_pytorch, 4)
    print("no mask: precision {}, recall {}, accuracy {}, f1 {}".format(precision, recall, accuracy, f1))

    precision, recall, accuracy,f1 = classify.evaluate_using_confusion_matrix(list_path_mask, database_embedding_facenet_pytorch, 4)
    print("mask: precision {}, recall {}, accuracy {}, f1 {}".format(precision, recall, accuracy, f1))

if __name__ == "__main__":
    main()