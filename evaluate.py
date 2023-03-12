import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
import csv
import tensorflow as tf
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,precision_recall_fscore_support
from train_tensorflow.models import convert_model_to_embedding, call_instance_model,call_instance_model_old
from train_tensorflow.Classify import Classify
from scipy.spatial.distance import cosine,euclidean
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm.auto import tqdm
from facenet_pytorch import MTCNN
from train_pytorch.detect_face import FaceDetector
from product.ModelController import ModelController
from product.FaceNetPytorch import FaceNetPytorch
from tool.FormatFunction import FormatFunction
from tool.FileFunction import FileFunction
from tool.GlobalValue import GlobalValue


def evaluate_lfw(model_controller, distance_type = "euclidean"):
    # setup
    my_model = ModelController("NewFacenet")
    # my_model = FaceNetPytorch("casia-webface")
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
    pairs = lfw.pairs
    labels = lfw.target
    face_detector = FaceDetector()
    
    #Loop different threshold
    thresholds = np.arange(0,2.01,0.01)
    predictions = [[] for i in range(len(thresholds))]
    for i in tqdm(range(0, pairs.shape[0])):
        pair = pairs[i]
        img1 = face_detector.detect_one_face(pair[0], optimize_margin = True)
        img2 = face_detector.detect_one_face(pair[1], optimize_margin = True)
        prediction = my_model.verify(img1, img2, thresholds, distance_type = distance_type) #this should return 1 for same person, 0 for different persons.
        for j in range(len(prediction)):
            predictions[j].append(prediction[j])
    scores = []
    for prediction in predictions:
        score = accuracy_score(labels, prediction)*100
        scores.append(score)
    for j in range(len(scores)):
        print(f"threshold {thresholds[j]} accuracy_score {scores[j]}")

def evaluate_mask_lfw():
    #Init
    model_name = "InceptionResNetV2"
    last_layer = "Dense"
    MODEL_NAME = f"160-64-{model_name}-{last_layer}"
    distance_formula = euclidean
    global_value = GlobalValue(image_size=[160,160], batch_size = 64, shuffle_size = 512, ratio_train = 0.8, epochs = 40, small_epochs = 2)
    format_function = FormatFunction(global_value)
    file_function = FileFunction()

    for i in range(40,41):
        print("evaluate epoch: ", i)
        path_model = os.path.join(os.getcwd(),"save_model",
                                  MODEL_NAME,f"epoch{i}.h5")
        print(path_model)
        model = call_instance_model((global_value.IMAGE_SIZE[0], global_value.IMAGE_SIZE[1],3), 12593, 512,last_layer, model_name)
        model.load_weights(path_model)
        model = convert_model_to_embedding(model, cut_position=-3)
        model_controller = ModelController(model = model)
        classify = Classify(model_controller, format_function)
        tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
        
        #Get embedding database, get no-mask face image then convert to vector
        print("embedding")
        encoding_path = os.path.join(os.getcwd(), "cache", "encodings", MODEL_NAME, f"epoch{i}.pkl")
        if not os.path.exists(encoding_path):
            database_embedding = classify.embedding_all_data_by_directory(
                os.path.join(os.getcwd(),"dataset","lfw_align"))
            classify.save_embedding_to_file(database_embedding, encoding_path)
        else:
            database_embedding = classify.load_embedding_from_file(encoding_path)
        
        for label, vector in database_embedding.items():
            # database_embedding[label] = vector/np.linalg.norm(vector)
            print(label, vector)
            
        # Preprocess data, get path to image with and without mask
        # print("predict")
        # print(distance_formula)
        # paths = list()
        # mask_data_directory = os.path.join(os.getcwd(), "dataset", "lfw_mask")
        # paths.extend(file_function.getPath(mask_data_directory))
        # # no_mask_data_directory = os.path.join(os.getcwd(), "dataset", "lfw_align")
        # # paths.extend(file_function.getPath(no_mask_data_directory))
        # # Accuracy
        # precision, recall, accuracy, f1 = classify.evaluate_using_confusion_matrix(paths, database_embedding, 
        #                                                     thresh_hold=4, distance_formula=distance_formula)
        # with open(os.path.join(os.getcwd(),"cache","metrics",MODEL_NAME+".csv"), "a",  newline =  '') as f:
        #     row = [i, precision, recall, accuracy, f1]
        #     writer = csv.writer(f)
        #     writer.writerow(row)


evaluate_mask_lfw()