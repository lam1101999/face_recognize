from train_tensorflow.FaceNet import call_instance_FaceNet_with_last_ArcFace, convert_arcface_model_to_embedding,\
                                        convert_arcface_model_to_embeddingv2,call_instance_FaceNet_with_last_isDense,\
                                        convert_train_model_to_embedding
import numpy as np
import os
from scipy.spatial.distance import euclidean, cosine
import cv2
class ModelController:
    def __init__(self, model_name = None, model = None):
        self.model_name = model_name
        if model_name is not None:
            self.model = self._init_model(self.model_name)
        elif model is not None:
            self.model = model
            
    def get_model(self):
        return self.model
    
    def _init_model(self, model_name = "Facenet"):
        model = None
        if model_name == "Facenet":
            path_to_weight = f"G:\My Drive\Colab Notebooks\FaceMaskRecognize\models\epoch49.h5"
            face_net_model = call_instance_FaceNet_with_last_isDense(
            input_shape=[110, 110, 3], number_of_class=10575, embedding=128)
            face_net_model.load_weights(path_to_weight)
            face_net_model = convert_train_model_to_embedding(face_net_model)
            model = face_net_model
        if model_name == "NewFacenet":
            path_to_weight = f"G:\My Drive\Colab Notebooks\FaceMaskRecognize\save_model\\110-ASIAN\epoch54.h5"
            print(path_to_weight)
            face_net_model = call_instance_FaceNet_with_last_isDense(
            input_shape=[110, 110, 3], number_of_class=12593, embedding=128)
            face_net_model.load_weights(path_to_weight)
            face_net_model = convert_train_model_to_embedding(face_net_model)
            model = face_net_model
        if model_name == "ArcFace":
            path_to_weight = f"G:\My Drive\Colab Notebooks\FaceMaskRecognize\models\epoch101.h5"
            arc_face_model = call_instance_FaceNet_with_last_ArcFace(
            input_shape=[110, 110, 3], number_of_class = 12593, embedding=512)
            arc_face_model.load_weights(path_to_weight)
            arc_face_model = convert_arcface_model_to_embedding(arc_face_model)
            model = arc_face_model
        
        if model == None :
            raise Exception("model is not support")
        else:
            return model
    
    def represent(self, image):
        size = 110
        print(image.shape)
        if self.model_name == "Facenet":
            size = 110
        if self.model_name == "ArcFace":
            size = 110
        image = cv2.resize(image,[size,size])
        vector_image = self.model(np.expand_dims(image, 0))[0]
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
    model_controller = ModelController("NewFacenet")
if __name__ =="__main__":
    main()