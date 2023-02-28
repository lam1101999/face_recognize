
from cmath import pi
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.losses import metric_learning
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Flatten, Dense, Layer, Conv2D, LeakyReLU, Concatenate,\
    Lambda, Add, add, MaxPooling2D, GlobalAveragePooling2D
from typing import Union, Callable
from train_tensorflow.inceptionresnetv1 import InceptionResNetV1
from train_tensorflow.inceptionresnetv1 import ArcFace
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import math


class LayerBeforeArcFace(tf.keras.layers.Layer):
    def __init__(self, num_classes, regularizer=regularizers.L2(1e-4), name=""):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.regularizer = regularizer

    def build(self, input_shape):
        embedding_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self.num_classes),
                                  dtype = tf.float32,
                                  initializer=tf.keras.initializers.HeNormal(),
                                  regularizer=self.regularizer,
                                  trainable=True,
                                  name='cosine_weights')
        self.built = True
        
    @tf.function
    def call(self, inputs):

        embedding = inputs
        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        return cosine_sim


class ArcFaceLoss(tf.keras.losses.Loss):
    # Modify from "https://github.com/yinguobing/arcface/blob/main/losses.py"
    def __init__(self,
                 scale=30.0,
                 margin=0.5,
                 name='arcface',
                 **kwargs):
        super().__init__(name=name)
        self.scale = float(scale)
        self.margin = float(margin)

    @tf.function
    def call(self, y_true, y_predict):
        # Convert y_true format array([ground truth number]) to format one-hot coding
        y_true = tf.reshape(y_true, [-1])
        depth = tf.shape(y_predict)[-1]
        y_true_one_hot = tf.one_hot(y_true,depth)
        # Prevent nan value
        # Calculate arcos from cos
        arc_cosine = tf.acos(K.clip(y_predict, -1 + K.epsilon(), 1 - K.epsilon()))
        # Add constant factor m to the angle corresponding to the ground truth label
        arc_cosine_with_margin = arc_cosine + y_true_one_hot*self.margin
        #convert arc_cosine_margin to cosine
        cosine_with_margin_scale = tf.cos(arc_cosine_with_margin)*self.scale

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, cosine_with_margin_scale)
        losses = tf.math.reduce_mean(losses)
        return losses

def convert_model_to_embedding(train_model, cut_position = -2, add_normalization=False):
    outputs = train_model.layers[cut_position].output
    if add_normalization:
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    embedding = tf.keras.models.Model(
        train_model.input, outputs, name="embedding")
    return embedding

def convert_dense_layer_to_arcface(path_weights, input_shape, number_of_class, embedding, model_name = "InceptionResNetV1", add_normalization = False):
    model = call_instance_model(input_shape, number_of_class, embedding, model_name, "Dense")
    model.load_weights(path_weights)
    outputs = model.layers[-2].output
    if add_normalization:
        outputs = BatchNormalization(momentum=0.995, epsilon=0.001,
	                    scale=False, name='Bottleneck_BatchNorm')(outputs)
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    outputs = LayerBeforeArcFace(number_of_class,name = "Layer_Before_ArcFace")(outputs)
    arcface_model = tf.keras.models.Model(inputs = model.input,outputs = outputs)
    return arcface_model

def special_convert_dense_layer_to_arcface(path_weights, input_shape, number_of_class, embedding, model_name = "InceptionResNetV1", add_normalization = False):
    model = call_instance_model(input_shape, number_of_class, embedding, model_name, "ArcFace")
    model.load_weights(path_weights)
    outputs = model.layers[-3].output
    outputs = Dropout(0.2, name='Dropout')(outputs)
    outputs = model.layers[-2](outputs)
    if add_normalization:
        outputs = BatchNormalization(momentum=0.995, epsilon=0.001,
	                    scale=False, name='Bottleneck_BatchNorm')(outputs)
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    outputs = LayerBeforeArcFace(number_of_class,name = "Layer_Before_ArcFace")(outputs)
    arcface_model = tf.keras.models.Model(inputs = model.input, outputs = outputs)
    return arcface_model

def call_instance_model(input_shape, number_of_class, embedding, model_name = "InceptionResNetV1",\
                        last_layer = "Dense"):
    
    # Create model structure
    embedding_model = None
    if model_name == "InceptionResNetV1":
        embedding_model = InceptionResNetV1(input_shape, embedding)
    if model_name == "InceptionResNetV1Hard":
        embedding_model = InceptionResNetV1(input_shape, embedding, easy_version = False)
    elif model_name == "InceptionResNetV2":
        embedding_model = tf.keras.applications.InceptionResNetV2(include_top=True, weights=None,\
                        input_shape=input_shape, classes = embedding, classifier_activation=None)
    elif model_name == "EfficientNetV2M":
        embedding_model = tf.keras.applications.EfficientNetV2M(include_top=True, weights=None,\
                        input_shape=input_shape, classes = embedding, classifier_activation=None)
        
    # Create last layer
    outputs = None
    if last_layer == "Dense":
        outputs = tf.keras.layers.Dense(
        number_of_class, use_bias=False, name='Bottleneck_train')(embedding_model.output)
    elif last_layer == "ArcFace":
        outputs = LayerBeforeArcFace(number_of_class,name = "Layer_Before_ArcFace")(embedding_model.output)
    
    # Create model from input and output
    model = tf.keras.models.Model(
        embedding_model.input, outputs, name=model_name)
    return model

if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    model_name = "InceptionResNetV2"
    path_dense = os.path.join(os.path.dirname(os.getcwd()), "save_model"
                              ,"160-64-InceptionResNetV2-Dense",
                              "epoch24.h5")
    input_shape = [160,160,3]
    number_of_class = 12593
    face_net_model = convert_dense_layer_to_arcface(path_dense, input_shape,
                                                    number_of_class, 512, model_name = model_name,
                                                    add_normalization= True)
    face_net_model.summary()
