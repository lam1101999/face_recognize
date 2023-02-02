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
from train_tensorflow.FaceNet import call_instance_FaceNet_with_last_isDense, convert_train_model_to_embedding


class Classify:
    def __init__(self, model_controller, format_function):
        self.model_controller = model_controller
        self.file_function = FileFunction()
        self.format_function = format_function

    def init_model_controller(self, model_controller):
        self.model_controller = model_controller

    def embedding_all_data_by_directory(self, all_data_directory, probability_train=None, is_normalized=False) -> dict:
        """ Receive directory of all class then produce embedding of each class as a vector,
            using tf.dataset to processing image

        Args:
            all_data_directory (_type_): path to the directory
            probability_train (_type_, optional): _description_. Defaults to None.

        Returns:
            Dict: dictionary include in form "class_name: embedding"
        """
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)
        for each_folder in tqdm(path_to_each_folder, ascii=" *"):
            label = each_folder.split(os.path.sep)[-1]
            path_each_file = self.file_function.getPath(each_folder)
            if (probability_train is not None):
                number_file = len(path_each_file)
                min = np.maximum(int(probability_train*number_file), 1)
                path_each_file = path_each_file[:min]
            if len(path_each_file) <= 0:
                continue
            dataset_of_one_person = tf.data.Dataset.from_tensor_slices(
                path_each_file)
            dataset_of_one_person = dataset_of_one_person.map(
                self.format_function.get_label_as_string, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_of_one_person = dataset_of_one_person.map(
                self.format_function.process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset_of_one_person = dataset_of_one_person.batch(10)
            embedding_one_person = self.embedding_data_one_label(
                dataset_of_one_person, is_normalized)
            encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary

    def embedding_data_one_label(self, dataset, is_normalized=False):
        """ receive dataset of one class then convert them into embedding,
            return average value of all embedding

        Args:
            dataset (_type_): tf.dataset of image of one class

        Returns:
            _type_: _description_
        """
        encodes = self.model_controller.get_model().predict(dataset)
        if encodes.any():
            encodes = np.average(encodes, axis=0)
            if is_normalized:
                encodes = encodes/np.linalg.norm(encodes)
        return encodes

    def embedding_one_data_by_directory(self, one_data_directory, encoding_dictionary=dict(), probability_train=None, is_normalized=False):
        label = one_data_directory.split(os.path.sep)[-1]
        path_each_file = self.file_function.getPath(one_data_directory)
        if (probability_train is not None):
            number_file = len(path_each_file)
            min = np.maximum(int(probability_train*number_file), 1)
            path_each_file = path_each_file[:min]
        dataset_of_one_person = tf.data.Dataset.from_tensor_slices(
            path_each_file)
        dataset_of_one_person = dataset_of_one_person.map(
            self.format_function.get_label_as_string, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_of_one_person = dataset_of_one_person.map(
            self.format_function.process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_of_one_person = dataset_of_one_person.batch(10)
        embedding_one_person = self.embedding_data_one_label(
            dataset_of_one_person, is_normalized)
        encoding_dictionary[label] = embedding_one_person
        return encoding_dictionary

    def embedding_all_data_by_directoryV2(self, all_data_directory, probability_train=None):
        """ Receive directory of all class then produce embedding of each class as a vector,
            using Pillow to processing image

        Args:
            all_data_directory (_type_): path to the directory
            probability_train (_type_, optional): _description_. Defaults to None.

        Returns:
            Dict: dictionary include in form "class_name: embedding"
        """
        encoding_dictionary = dict()
        path_to_each_folder = self.file_function.getSubDir(all_data_directory)
        # work with image each person
        for each_folder in tqdm(path_to_each_folder, ascii=" *"):
            data = list()
            label = each_folder.split(os.path.sep)[-1]
            # get all path image of this person
            path_each_files = self.file_function.getPath(each_folder)
            if (probability_train is not None):
                number_file = len(path_each_files)
                min = np.maximum(int(probability_train*number_file), 1)
                path_each_files = path_each_files[:min]
            if len(path_each_files) <= 0:
                continue

            # Convert path image to image
            for path_each_file in path_each_files:
                try:
                    image = self.format_function.open_and_process_image_Pillow(
                        path_each_file)
                    data.append(image)
                except Exception:
                    continue

            data = np.asarray(data)
            # get embedding
            embedding_one_person = self.embedding_data_one_label(data)
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

    def detect_one_image(self, image, embedding, thresh_hold=4, distance_formula=cosine):
        start = time.time()
        image = np.expand_dims(image, 0)
        encode = self.model_controller.get_model().predict(image)[0]
        name = "unknown"
        distance = float("inf")
        for db_name, db_encode in embedding.items():
            dist = distance_formula(db_encode, encode)
            if dist < thresh_hold and dist < distance:
                name = db_name
                distance = dist
        return name

    def detect_on_dataset(self, dataset, embedding, thresh_hold=4, distance_formula=cosine):
        list_name = list()
        encode = self.model_controller.get_model().predict(dataset)
        for each_encode in tqdm(encode, ascii=" *"):
            name = "unknown"
            distance = float("inf")
            for db_name, db_encode in embedding.items():
                dist = distance_formula(db_encode, each_encode)
                if dist < thresh_hold and dist < distance:
                    name = db_name
                    distance = dist
            list_name.append(name)

        return list_name

    def evaluate(self, list_path, embedding, thresh_hold=4):

        total_encoding_duration = 0
        total_finding_duration = 0
        unknow_answer = 0
        right_answer = 0
        mis_answer = 0
        total_answer = len(list_path)

        for each_path in tqdm(list_path, ascii=" *"):
            actual_label = each_path.split(os.path.sep)[-2]
            image = self.format_function.open_and_process_image_Pillow(
                each_path)
            predicted_label, encoding_duration, finding_duration = self.detect_one_image(
                image, embedding, thresh_hold)
            if (predicted_label == "unknown"):
                unknow_answer += 1
            elif (predicted_label == actual_label):
                right_answer += 1
            else:
                mis_answer += 1

            total_encoding_duration += encoding_duration
            total_finding_duration += finding_duration
        print("database have {} vector, evaluate {} image, average_time_encoding {}, average_time_finding {}".format(
            len(embedding), total_answer, total_encoding_duration/total_answer, total_finding_duration/total_answer))
        return right_answer, unknow_answer, mis_answer, total_answer

    def evaluate_using_confusion_matrix(self, list_path, embedding, thresh_hold=4, distance_formula=cosine):
        matrix = self.calculate_confusion_matrix(
            list_path, embedding, thresh_hold, distance_formula)
        precision, recall, accuracy, f1 = self.calculate_precision_recall_accuracy_f1_from_matrix(
            matrix)
        return precision, recall, accuracy, f1

    def calculate_confusion_matrix(self, list_path, embedding, thresh_hold=4, distance_formula=cosine):
        # Initialize Confusion Matrix (row is real label, column is predict label so precision is column, recall is row)
        total_label = len(embedding)
        matrix = np.zeros((total_label, total_label))

        # give label an offset to work with matrix
        off_set = 0
        off_set_dictionary = dict()
        for key in embedding.keys():
            off_set_dictionary[key] = off_set
            off_set += 1
        print(len(list_path))
        # predict label
        for each_path in tqdm(list_path, ascii=" *"):
            real_label = each_path.split(os.path.sep)[-2]
            image = self.format_function.open_and_process_image_Pillow(
                each_path)
            predict_label = self.detect_one_image(
                image, embedding, thresh_hold, distance_formula)
            # Assign value to matrix
            offset_row = off_set_dictionary[real_label]
            offset_column = off_set_dictionary[predict_label]
            matrix[offset_row, offset_column] = matrix[offset_row, offset_column]+1

        return matrix

    def calculate_precision_recall_accuracy_f1_from_matrix(self, confusion_matrix):
        one_size_matrix = len(confusion_matrix)

        # calculate precision we will calculate precision of each label then get average, ignore that label if that label does not have prediction(sum column = 0)
        total_label_have_precision = 0
        total_precision = 0
        for i in range(one_size_matrix):
            total_value_predict_as_i = np.sum(confusion_matrix[:, i])
            total_value_i_predict_as_i = confusion_matrix[i, i]
            if total_value_predict_as_i != 0:
                precision_i = total_value_i_predict_as_i/total_value_predict_as_i
                total_precision += precision_i
                total_label_have_precision += 1
        precision = total_precision/total_label_have_precision

        # calculate recall we will calculate recall of each label then get average, ignore that label if that label does not have recall(sum row = 0)
        total_label_have_recall = 0
        total_recall = 0
        for i in range(one_size_matrix):
            total_value_is_i = np.sum(confusion_matrix[i])
            total_value_i_predict_as_i = confusion_matrix[i, i]
            if total_value_is_i != 0:
                recall_i = total_value_i_predict_as_i/total_value_is_i
                total_recall += recall_i
                total_label_have_recall += 1
        recall = total_recall/total_label_have_recall

        # calculate accuracy
        total_right_answers = 0
        for i in range(one_size_matrix):
            total_right_answers += confusion_matrix[i, i]
        total_answers = np.sum(confusion_matrix)
        accuracy = total_right_answers/total_answers

        # calculate f1
        f1 = 2*(precision*recall)/(precision+recall)

        return precision, recall, accuracy, f1

    def measure_time_to_extract_embedding(self, directory_path):
        image_paths = self.file_function.getPath(directory_path)
        now  = time.time()
        for image_path in image_paths:
            image = self.format_function.open_and_process_image_Pillow(
                image_path)
            data = np.asarray(image)
            shape = (1, image.shape[0], image.shape[1], image.shape[2])
            data_in_array = np.reshape(data, shape)
            encodes = self.model_controller.get_model().predict(data_in_array)
        end = time.time()
        print(f"time to embedding one image {(end - now)/len(image_paths)}")


def main():
    global_value = GlobalValue(image_size=[110, 110], batch_size=512, shuffle_size=1000, ratio_train=0.8, ratio_test=0.1,
        ratio_valid=0.1, epochs=40, small_epochs=50, image_each_class=15)
    format_function = FormatFunction(global_value)
    model_path = os.path.join(
        os.path.dirname(os.getcwd()), "save_model", "110-ASIAN", "epoch54.h5")
    input_size = [110, 110, 3]
    reload_model = call_instance_FaceNet_with_last_isDense(
        input_size, 12593, embedding=128)
    reload_model.load_weights(model_path)
    embedding_model = convert_train_model_to_embedding(reload_model)
    classify = Classify(embedding_model, format_function)
    classify.measure_time_to_extract_embedding(os.path.join(os.path.dirname(os.getcwd()),"dataset","10_person"))


if __name__ == "__main__":
    main()
