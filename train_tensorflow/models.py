'Function to call model'

import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.losses import metric_learning
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Layer, Conv2D, LeakyReLU, Concatenate,\
    Lambda, Add, add, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    InceptionResNetV2
)
from train_tensorflow.inceptionresnetv1 import InceptionResNetV1
from tensorflow.keras import regularizers
import math


class LayerBeforeArcFace(tf.keras.layers.Layer):
    def __init__(self, num_classes, regularizer=regularizers.L2(1e-4), name=""):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.regularizer = regularizer

    def build(self, input_shape):
        embedding_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self.num_classes),
                                  dtype=tf.float32,
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
        y_true_one_hot = tf.one_hot(y_true, depth)
        # Prevent nan value
        # Calculate arcos from cos
        arc_cosine = tf.acos(
            K.clip(y_predict, -1 + K.epsilon(), 1 - K.epsilon()))
        # Add constant factor m to the angle corresponding to the ground truth label
        arc_cosine_with_margin = arc_cosine + y_true_one_hot*self.margin
        # convert arc_cosine_margin to cosine
        cosine_with_margin_scale = tf.cos(arc_cosine_with_margin)*self.scale

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            y_true, cosine_with_margin_scale)
        losses = tf.math.reduce_mean(losses)
        return losses


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convert_model_to_embedding(train_model, cut_position=-2, add_normalization=False):
    outputs = train_model.layers[cut_position].output
    if add_normalization:
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    embedding = tf.keras.models.Model(
        train_model.input, outputs, name="embedding")
    return embedding


def convert_dense_layer_to_arcface(path_weights, input_shape, number_of_class, embedding, model_name="InceptionResNetV1", add_normalization=False):
    model = call_instance_model(
        input_shape, number_of_class, embedding, model_name, "Dense")
    model.load_weights(path_weights)
    outputs = model.layers[-2].output
    if add_normalization:
        outputs = BatchNormalization(momentum=0.995, epsilon=0.001,
                                     scale=False, name='Bottleneck_BatchNorm')(outputs)
        outputs = tf.keras.layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1))(outputs)
    outputs = LayerBeforeArcFace(
        number_of_class, name="Layer_Before_ArcFace")(outputs)
    arcface_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
    return arcface_model


def special_convert_dense_layer_to_arcface(path_weights, input_shape, number_of_class, embedding, model_name="InceptionResNetV1"):
    model = call_instance_model(
        input_shape, number_of_class, embedding, model_name, "Dense")
    model.load_weights(path_weights)
    outputs = model.layers[-3].output
    outputs = Dropout(0.2, name='Dropout')(outputs)
    outputs = model.layers[-2](outputs)
    outputs = BatchNormalization(momentum=0.995, epsilon=0.001,
                                 scale=False, name='Bottleneck_BatchNorm')(outputs)
    # outputs = LayerBeforeArcFace(number_of_class,name = "Layer_Before_ArcFace")(outputs)
    outputs = model.layers[-1](outputs)
    arcface_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
    return arcface_model


def call_instance_model(input_shape, num_classes=None, embd_shape=512, head_type=None,
                        backbone_type="InceptionResNetV1", use_pretrain=True, name="facenet"):
    x = inputs = Input(input_shape, name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape)(x)

    # Create last layer
    if head_type is not None:
        assert num_classes is not None
        if head_type == "ArcFace":
            logist = LayerBeforeArcFace(number_of_class=num_classes)(embds)
        elif head_type == "Dense":
            logist = Dense(num_classes)(embds)
        return Model(inputs, logist, name=name)
    else:
        return Model(inputs, embds, name=name)


def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'
    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return  ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == "InceptionResNetV2":
            return InceptionResNetV2(input_shape=x_in.shape[1:], include_top=False,
                                    weights=weights)(x_in)
        else:
            raise TypeError('backbone_type error!')
    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer="l2")(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


if __name__ == "__main__":
    # tf.config.run_functions_eagerly(True)
    model_name = "InceptionResNetV2"
    input_shape = [160, 160, 3]
    number_of_class = 12593

    model = call_instance_model(
        [160, 160, 3], number_of_class, 512, model_name, "ArcFace")
    model = convert_model_to_embedding(model)
    model.summary()
