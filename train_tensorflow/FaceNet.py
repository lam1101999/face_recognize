
from cmath import pi
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.losses import metric_learning
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Flatten, Dense, Layer, Conv2D, LeakyReLU, Concatenate,\
    Lambda, Add, add, MaxPooling2D, GlobalAveragePooling2D
from typing import Union, Callable
from train_tensorflow.Net import InceptionResNetV1
from train_tensorflow.Net import ArcFace
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from keras.utils.layer_utils import count_params
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


def call_instance_FaceNet_with_last_isDense(input_shape, number_of_class, embedding):
    embedding_model = InceptionResNetV1(input_shape, embedding)
    # The face-net model
    outputs = tf.keras.layers.Dense(
        number_of_class, use_bias=False, name='Bottleneck_train')(embedding_model.output)
    face_net_model = tf.keras.Model(
        embedding_model.input, outputs, name="FaceNetModel")
    return face_net_model



def convert_train_model_to_embedding(train_model):
    """  convert train model to embedding with batch normalization

    Args:
        train_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    cut_the_last_layer = tf.keras.models.Model(
        inputs=train_model.input, outputs=train_model.layers[-3].output)
    outputs = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1))(cut_the_last_layer.output)
    face_net_model = tf.keras.Model(
        cut_the_last_layer.input, outputs, name="FaceNetModel")
    return face_net_model

def convert_train_model_to_embeddingv2(train_model):
    """convert train model to embedding without batch normalization

    Args:
        train_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    cut_the_last_layer = tf.keras.models.Model(
        inputs=train_model.input, outputs=train_model.layers[-2].output)
    outputs = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1))(cut_the_last_layer.output)
    face_net_model = tf.keras.Model(
        cut_the_last_layer.input, outputs, name="FaceNetModel")
    return face_net_model
def convert_arcface_model_to_embedding(train_model):
    bottle_neck = tf.keras.models.Model(
        inputs=train_model.input, outputs=train_model.layers[-3].output)
    face_net_model = tf.keras.Model(
        bottle_neck.input, bottle_neck.output, name="FaceNetModel")
    return face_net_model

def convert_arcface_model_to_embeddingv2(train_model):
    bottle_neck = tf.keras.models.Model(
        inputs=train_model.input, outputs=train_model.layers[-2].output)
    face_net_model = tf.keras.Model(
        bottle_neck.input, bottle_neck.output, name="FaceNetModel")
    return face_net_model


def call_instance_FaceNet_with_last_ArcFace(input_shape, number_of_class, embedding):
    embedding_model = InceptionResNetV1(input_shape, embedding)
    # The face-net model
    outputs = LayerBeforeArcFace(number_of_class,name = "Layer_Before_ArcFace")(embedding_model.output)
    model = tf.keras.Model(
        embedding_model.input, outputs, name="FaceNetModel")
    return model

def call_instance_model(input_shape, number_of_class, embedding, model_name = "InceptionResNetV1",\
                        last_layer = "Dense"):
    
    # Create model structure
    embedding_model = None
    if model_name == "InceptionResNetV1":
        embedding_model = InceptionResNetV1(input_shape, embedding)
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
    model = tf.keras.Model(
        embedding_model.input, outputs, name=model_name)
    return model

if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    input_shape = (128, 128, 3)
    number_of_classes = 1000
    model = call_instance_FaceNet_with_last_isDense(input_shape,10575,128)
    model.summary()
