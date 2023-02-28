import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense,\
                                Dropout, BatchNormalization, Concatenate, Lambda, add, GlobalAveragePooling2D, \
                                   Convolution2D, LocallyConnected2D, ZeroPadding2D, concatenate, AveragePooling2D, \
                                Layer, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers


def scaling(x, scale):
    return x * scale


def InceptionResNetV1(input_shape, embedding, easy_version = True ):

    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, strides=2, padding='valid',
               use_bias=False, name='Conv2d_1a_3x3')(inputs)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                           scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
    x = LeakyReLU(name='Conv2d_1a_3x3_Activation')(x)
    x = Conv2D(32, 3, strides=1, padding='valid',
               use_bias=False, name='Conv2d_2a_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                           scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
    x = LeakyReLU(name='Conv2d_2a_3x3_Activation')(x)
    x = Conv2D(64, 3, strides=1, padding='same',
               use_bias=False, name='Conv2d_2b_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                           scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
    x = LeakyReLU(name='Conv2d_2b_3x3_Activation')(x)
    x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
    x = Conv2D(80, 1, strides=1, padding='valid',
               use_bias=False, name='Conv2d_3b_1x1')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                           scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
    x = LeakyReLU(name='Conv2d_3b_1x1_Activation')(x)
    x = Conv2D(192, 3, strides=1, padding='valid',
               use_bias=False, name='Conv2d_4a_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                           scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
    x = LeakyReLU(name='Conv2d_4a_3x3_Activation')(x)
    x = Conv2D(256, 3, strides=2, padding='valid',
               use_bias=False, name='Conv2d_4b_3x3')(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                           scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
    x = LeakyReLU(name='Conv2d_4b_3x3_Activation')(x)

    # 5x Block35 (Inception-ResNet-A block):
    x = A_Block(x, "A_BLOCK_1_")
    x = A_Block(x, "A_BLOCK_2_")
    x = A_Block(x, "A_BLOCK_3_")
    x = A_Block(x, "A_BLOCK_4_")
    x = A_Block(x, "A_BLOCK_5_")

    # Mixed 6a (Reduction-A block):
    branch_0 = Conv2D(384, 3, strides=2, padding='valid',
                      use_bias=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3')(x)
    branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
    branch_0 = LeakyReLU(
        name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
    branch_1 = Conv2D(192, 1, strides=1, padding='same',
                      use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1')(x)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
    branch_1 = LeakyReLU(
        name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
    branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False,
                      name='Mixed_6a_Branch_1_Conv2d_0b_3x3')(branch_1)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
    branch_1 = LeakyReLU(
        name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
    branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                      name='Mixed_6a_Branch_1_Conv2d_1a_3x3')(branch_1)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
    branch_1 = LeakyReLU(
        name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
    branch_pool = MaxPooling2D(
        3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=3, name='Mixed_6a')(branches)

    # 10x Block17 (Inception-ResNet-B block):
    x = B_Block(x, "B_BLOCK_1_")
    x = B_Block(x, "B_BLOCK_2_")
    x = B_Block(x, "B_BLOCK_3_")
    x = B_Block(x, "B_BLOCK_4_")
    x = B_Block(x, "B_BLOCK_5_")
    x = B_Block(x, "B_BLOCK_6_")
    x = B_Block(x, "B_BLOCK_7_")
    x = B_Block(x, "B_BLOCK_8_")
    x = B_Block(x, "B_BLOCK_9_")
    x = B_Block(x, "B_BLOCK_10_")

    # Mixed 7a (Reduction-B block):
    branch_0 = Conv2D(256, 1, strides=1, padding='same',
                      use_bias=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1')(x)
    branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
    branch_0 = LeakyReLU(
        name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
    branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False,
                      name='Mixed_7a_Branch_0_Conv2d_1a_3x3')(branch_0)
    branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
    branch_0 = LeakyReLU(
        name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
    branch_1 = Conv2D(256, 1, strides=1, padding='same',
                      use_bias=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1')(x)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
    branch_1 = LeakyReLU(
        name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
    branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                      name='Mixed_7a_Branch_1_Conv2d_1a_3x3')(branch_1)
    branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
    branch_1 = LeakyReLU(
        name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
    branch_2 = Conv2D(256, 1, strides=1, padding='same',
                      use_bias=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1')(x)
    branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
    branch_2 = LeakyReLU(
        name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
    branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False,
                      name='Mixed_7a_Branch_2_Conv2d_0b_3x3')(branch_2)
    branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
    branch_2 = LeakyReLU(
        name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
    branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                      name='Mixed_7a_Branch_2_Conv2d_1a_3x3')(branch_2)
    branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                  scale=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
    branch_2 = LeakyReLU(
        name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
    branch_pool = MaxPooling2D(
        3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=3, name='Mixed_7a')(branches)

    # 5x Block8 (Inception-ResNet-C block):
    x = C_Block(x, "C_BLOCK_1_")
    x = C_Block(x, "C_BLOCK_2_")
    x = C_Block(x, "C_BLOCK_3_")
    x = C_Block(x, "C_BLOCK_4_")
    x = C_Block(x, "C_BLOCK_5_")

    # Classification block
    x = GlobalAveragePooling2D(name='AvgPool')(x)
    if not easy_version:
        x = Dropout(0.2, name='Dropout')(x)
    # Bottleneck
    x = Dense(embedding, use_bias=False, name='Bottleneck')(x)
    if not easy_version:
        x = BatchNormalization(momentum=0.995, epsilon=0.001,
                        scale=False, name='Bottleneck_BatchNorm')(x)
    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')

    return model


def A_Block(inputs, name="A_BLOCK"):
    with tf.name_scope(name):
        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False,
                          name=name + 'Block35_Branch_0_Conv2d_1x1')(inputs)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name + 'Block35_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = LeakyReLU(
            name=name + 'Block35_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False,
                          name=name + 'Block35_Branch_1_Conv2d_0a_1x1')(inputs)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name + 'Block35_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name + 'Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name=name + 'Block35_Branch_1_Conv2d_0b_3x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name + 'Block35_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name + 'Block35_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False,
                          name=name + 'Block35_Branch_2_Conv2d_0a_1x1')(inputs)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name + 'Block35_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = LeakyReLU(
            name=name + 'Block35_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name=name + 'Block35_Branch_2_Conv2d_0b_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name + 'Block35_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = LeakyReLU(
            name=name + 'Block35_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False,
                          name=name + 'Block35_Branch_2_Conv2d_0c_3x3')(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name + 'Block35_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = LeakyReLU(
            name=name + 'Block35_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name=name + 'Block35_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True,
                    name=name + 'Block35_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={
                    'scale': 0.17}, name=name + "Scaling")(up)
        inputs = add([inputs, up], name=name + "Addding")
        inputs = LeakyReLU(name=name + 'Block35_Activation')(inputs)
    return inputs


def Reduction_A_Block(input, name):
    pass


def B_Block(inputs, name="B_BLOCK"):
    with tf.name_scope(name):
        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False,
                          name=name+'Block17_Branch_0_Conv2d_1x1')(inputs)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name=name+'Block17_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = LeakyReLU(
            name=name+'Block17_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False,
                          name=name+'Block17_Branch_1_Conv2d_0a_1x1')(inputs)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name+'Block17_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name+'Block17_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name=name+'Block17_Branch_1_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name+'Block17_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name+'Block17_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name=name+'Block17_Branch_1_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name+'Block17_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name+'Block17_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name=name+'Block17_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True,
                    name=name+'Block17_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={
                    'scale': 0.1}, name=name + "Scaling")(up)
        inputs = add([inputs, up], name=name + "Addding")
        inputs = LeakyReLU(name=name+'Block17_Activation')(inputs)
    return inputs


def C_Block(inputs, name="C_BLOCK"):
    with tf.name_scope(name):
        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False,
                          name=name+'Block8_1_Branch_0_Conv2d_1x1')(inputs)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                      scale=False, name=name+'Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = LeakyReLU(
            name=name+'Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False,
                          name=name+'Block8_1_Branch_1_Conv2d_0a_1x1')(inputs)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name+'Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name+'Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name=name+'Block8_1_Branch_1_Conv2d_0b_1x3')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name+'Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name+'Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name=name+'Block8_1_Branch_1_Conv2d_0c_3x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name=name+'Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = LeakyReLU(
            name=name+'Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name=name+'Block8_1_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True,
                    name=name+'Block8_1_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={
                    'scale': 0.2}, name=name + "Scaling")(up)
        inputs = add([inputs, up], name=name + "Addding")
        inputs = LeakyReLU(name=name+'Block8_1_Activation')(inputs)
    return inputs


def _resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return tf_utils.constant_value(training)


class ArcFace(Layer):
    """
    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698
    https://www.kaggle.com/code/chankhavu/keras-layers-arcface-cosface-adacos

    Arguments:
      num_classes: number of classes to classify
      s: scale factor
      m: margin
      regularizer: weights regularizer
    """

    def __init__(self,
                 num_classes,
                 s=30.0,
                 m=0.5,
                 regularizer=None,
                 name='arcface',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer,
                                  name='cosine_weights')

    def call(self, inputs, training=None):
        """
        During training, requires 2 inputs: embedding (after backbone+pool+dense),
        and ground truth labels. The labels should be sparse (and use
        sparse_categorical_crossentropy as loss).
        """
        embedding, label = inputs

        # Squeezing is necessary for Keras. It expands the dimension to (n, 1)
        label = tf.reshape(label, [-1], name='label_shape_correction')

        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(embedding, axis=1, name='normalize_prelogits')
        w = tf.nn.l2_normalize(self._w, axis=0, name='normalize_weights')
        cosine_sim = tf.matmul(x, w, name='cosine_similarity')

        training = resolve_training_flag(self, training)
        if not training:
            # We don't have labels if we're not in training mode
            return self._s * cosine_sim
        else:
            one_hot_labels = tf.one_hot(label,
                                        depth=self._n_classes,
                                        name='one_hot_labels')
            theta = tf.math.acos(K.clip(
                    cosine_sim, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            selected_labels = tf.where(tf.greater(theta, math.pi - self._m),
                                       tf.zeros_like(one_hot_labels),
                                       one_hot_labels,
                                       name='selected_labels')
            final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                                   theta + self._m,
                                   theta,
                                   name='final_theta')
            output = tf.math.cos(final_theta, name='cosine_sim_with_margin')
            return self._s * output


if __name__ == "__main__":
    input = (100, 100, 3)
    model = InceptionResNetV1(input, 128)
    model.summary()