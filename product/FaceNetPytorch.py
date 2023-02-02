import numpy as np
import os
from scipy.spatial.distance import euclidean, cosine
import cv2
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.datasets import fetch_lfw_pairs
class FaceNetPytorch:
    def __init__(self, name = None):
        self.name = name
        self.model = self._init_model(self.name)
        self.transform = transforms.ToTensor()
    
    def _init_model(self, name = "casia-webface"):
        model = None
        if name == "casia-webface":
            model = InceptionResnetV1(pretrained = "casia-webface").eval()
        if name == "vggface2":
            model = InceptionResnetV1(pretrained = "vggface2").eval()
        
        if model == None:
            raise Exception("model is not support")
        else:
            return model
    
    def represent(self, image):
        size = 160
        image = cv2.resize(image,[size,size])
        image = self.transform(image).float()
        vector_image = self.model(image.unsqueeze(0)).detach()[0]
        return vector_image
    
    def calculate_distance(self, image_1, image_2, distance_type = "euclidean"):
        if distance_type == "euclidean":
            distance_type = euclidean
        if distance_type == "cosine":
            distance_type = cosine
        vector_1 = self.represent(image_1)
        vector_2 = self.represent(image_2)
        result = distance_type(vector_1,vector_2)
        return result
    
    def verify(self, image_1, image_2, thresholds, distance_type = "euclidean"):
        """This funciton verify if 2 image are the same people, thresholds have form array
        if there are more than one threshold
        

        Args:
            image_1 (_type_): _description_
            image_2 (_type_): _description_
            thresholds (_type_): _description_
            distance_type (str, optional): _description_. Defaults to "euclidean".

        Returns:
            _type_: _description_
        """
        distance = self.calculate_distance(image_1, image_2, distance_type)
        prediction = []
        for threshold in thresholds:
            if distance <= threshold:
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction

def main():
    lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
    pairs = lfw.pairs
    pair = pairs[0]
    image1 = pair[0]
    my_model = FaceNetPytorch("casia-webface")
    embe = my_model.represent(image1)
if __name__ =="__main__":
    main()