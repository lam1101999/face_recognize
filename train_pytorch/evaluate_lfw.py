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
from facenet_pytorch import MTCNN, InceptionResnetV1
from train_pytorch.detect_face import FaceDetector
from torchvision import transforms



def verify(face_net_model, image1, image2, thresholds):
    image1 = cv2.resize(image1, [110, 110])
    image2 = cv2.resize(image2, [110, 110])
    transform = transforms.ToTensor()
    image1 = transform(image1).float()
    image2 = transform(image2).float()
    
    vector_image_1 = face_net_model(image1.unsqueeze(0)).detach()[0]
    vector_image_2 = face_net_model(image2.unsqueeze(0)).detach()[0]
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
    model = InceptionResnetV1(pretrained='vggface2').eval()

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
        prediction = verify(model,img1, img2, thresholds) #this should return 1 for same person, 0 for different persons.
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
