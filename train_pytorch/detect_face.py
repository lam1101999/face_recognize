from facenet_pytorch import MTCNN
from torchvision import transforms
import cv2
import numpy as np

class FaceDetector():
    def __init__(self):
        self.mtcnn = MTCNN(margin=36, select_largest=False,
                  post_process=False)
        self.to_pil_image = transforms.ToPILImage()
    
    def detect_one_face(self,image, margin = 0, optimize_margin = False):
        margin_height = margin
        margin_width = margin
        image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        image = image.astype(np.uint8)
        width = np.shape(image)[1]
        height = np.shape(image)[0]
        if optimize_margin:
            margin_height = height*(10/100)
            margin_width = width*(10/100)
            
        boxes, probs = self.mtcnn.detect(image, landmarks=False)
        best_box = boxes[self._find_best_face_index(boxes)]
        left = np.maximum(0, int(best_box[0]-margin_width/2))
        top  = np.maximum(0, int(best_box[1]-margin_height/2))
        right  = np.minimum(width, int(best_box[2]+margin_width/2))
        bottom  = np.minimum(height, int(best_box[3]+margin_height/2))
        
        face = image[top:bottom,left:right]
        return face/255
    
    def _find_best_face_index(self, boxes):
        best_index = 0
        best_ratio = np.inf
        for i in range(len(boxes)):
            box = boxes[i]
            width = box[2] - box[0]
            height = box[3] - box[1]
            ratio = height/width
            if ratio < 1:
                raio = 1/ratio

            if ratio < best_ratio:
                best_index = i
        return best_index