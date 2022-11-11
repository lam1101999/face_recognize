import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
from sklearn.datasets import fetch_lfw_pairs
from train_tensorflow.FaceNet import call_instance_FaceNet_with_last_ArcFace, convert_arcface_model_to_embedding,\
                                        convert_arcface_model_to_embeddingv2,call_instance_FaceNet_with_last_isDense,\
                                        convert_train_model_to_embedding
from scipy.spatial.distance import cosine,euclidean
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from facenet_pytorch import MTCNN
from train_pytorch.detect_face import FaceDetector
from torchvision import transforms


def verify(face_net_model, image1, image2, thresholds):
    image1 = cv2.resize(image1, [110, 110])
    image2 = cv2.resize(image2, [110, 110])

    vector_image_1 = face_net_model(np.expand_dims(image1, 0))[0]
    vector_image_2 = face_net_model(np.expand_dims(image2, 0))[0]
    distance = euclidean(vector_image_1, vector_image_2)
    prediction = []
    for threshold in thresholds:
        if distance <= threshold:
            prediction.append(1)
        else:
            prediction.append(0)
    return prediction

def main():
    # Prepare model
    path = os.path.join(os.path.dirname(os.getcwd()),
                        "cache", "data", "label_dict.pkl")
    with open(path, 'rb') as f:
        label_dict = pickle.load(f)
        
    # path_to_weight = os.path.join(os.path.dirname(
    #     os.getcwd()), "save_model", "110-64-ASIAN-ArcFace", "epoch101.h5")
    # face_net_model = call_instance_FaceNet_with_last_ArcFace(
    #     input_shape=[110, 110, 3], number_of_class=len(label_dict), embedding=512)
    
    path_to_weight = os.path.join(os.path.dirname(
        os.getcwd()), "save_model", "origin", "epoch49.h5")
    face_net_model = call_instance_FaceNet_with_last_isDense(
        input_shape=[110, 110, 3], number_of_class=10575, embedding=128)
    
    face_net_model.load_weights(path_to_weight)
    face_net_model = convert_train_model_to_embedding(face_net_model)

    # Setup
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
        prediction = verify(face_net_model,img1, img2, thresholds) #this should return 1 for same person, 0 for different persons.
        for j in range(len(prediction)):
            predictions[j].append(prediction[j])
    scores = []
    for prediction in predictions:
        score = accuracy_score(labels, prediction)
        scores.append(score)
    for j in range(len(scores)):
        print(f"threshold {thresholds[j]} accuracy_score {scores[j]}")


if __name__ == "__main__":
    main()
