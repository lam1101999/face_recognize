class GlobalValue:
    def __init__(self, image_size = None, batch_size = None, shuffle_size = None, ratio_train = None, ratio_test = None, ratio_valid = None, epochs = None,
     small_epochs = None, encoding_path = None, image_each_class = 10):
        #Init some variable
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size
        self.SHUFFLE_SIZE = shuffle_size
        self.RATIO_TRAIN = ratio_train
        self.RATIO_TEST = ratio_test
        self.RATIO_VALID = ratio_valid
        self.EPOCHS = epochs
        self.SMALL_EPOCHS = small_epochs
        self.ENCODING_PATH = encoding_path
        self.IMAGE_EACH_CLASS = image_each_class
        
    
    def init_mnist(self):
        #Init some variable for mnist dataset
        self.IMAGE_SIZE = [80,80]
        self.BATCH_SIZE = 56
        self.SHUFFLE_SIZE = self.BATCH_SIZE
        self.RATIO_TRAIN = 0.8
        self.RATIO_TEST = 1 - self.RATIO_TRAIN
        self.RATIO_VALID = 0
        self.EPOCHS = 40
        self.ENCODING_PATH = 'encodings/encodings.pkl'
        
    def init_face(self):
        #Init some variable for face datset
        self.IMAGE_SIZE = [100,100]
        self.BATCH_SIZE = 100
        self.SHUFFLE_SIZE = self.BATCH_SIZE
        self.RATIO_TRAIN = 0.8
        self.RATIO_TEST = 1 - self.RATIO_TRAIN
        self.RATIO_VALID = 0
        self.EPOCHS = 10
        self.SMALL_EPECHS = 10
        self.ENCODING_PATH = 'encodings/encodings.pkl'