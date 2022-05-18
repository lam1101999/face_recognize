import tensorflow
from wear_mask import MTCNN
import threading
import numpy as np
import os
class FaceDetector():
    def __init__(self):
        self.lock = threading.Lock()
        #Currently use code of anohter project with tensorflow v1...
        if tensorflow.__version__.startswith('1.'):
            import tensorflow as tf

        else:
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()

        self.GPU_ratio = 0.6
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        with tf.Graph().as_default():
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,
                                    )
            if self.GPU_ratio is None:
                config.gpu_options.allow_growth = True
            else:
                config.gpu_options.per_process_gpu_memory_fraction = self.GPU_ratio
            sess = tf.Session(config=config)
            with sess.as_default():
                self.pnet, self.rnet, self.onet = MTCNN.create_mtcnn(sess, None)
        
        os.system("cls")
    
    def detect_one_face(self, image):
        self.lock.acquire()
        try:
            boundingBoxes, _ = MTCNN.detect_face(image, self.minsize, self.pnet, self.rnet,
            self.onet,self.threshold, self.factor)
            boxCoordinates = boundingBoxes[0:1, 0:4]
            boxCoordinates = np.array(boxCoordinates)
            boxCoordinates = boxCoordinates.astype(np.int16)
        finally:
            self.lock.release()
            return boxCoordinates
    def get_coordinate_margin(self, bounding_box, margin, width, height):
        left = np.maximum(0, int(bounding_box[0]-margin/2))
        top  = np.maximum(0, int(bounding_box[1]-margin/2))
        right  = np.minimum(width, int(bounding_box[2]+margin/2))
        bottom  = np.minimum(height, int(bounding_box[3]+margin/2))
        return left,top,right,bottom