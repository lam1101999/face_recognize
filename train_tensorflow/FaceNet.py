
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


class FaceNetModel(tf.keras.Model):
    def __init__(self, number_of_class=None, arc_face=False):
        super().__init__()
        self.number_of_class = number_of_class
        self.arc_face = arc_face
        # Stem
        self.stem = Stem()

        # 5 A-Block
        self.a_block_1 = ABlock("A_BLOCK_1_")
        self.a_block_2 = ABlock("A_BLOCK_2_")
        self.a_block_3 = ABlock("A_BLOCK_3_")
        self.a_block_4 = ABlock("A_BLOCK_4_")
        self.a_block_5 = ABlock("A_BLOCK_5_")

        # Reduction A Block
        self.reduction_a_block = ReductionABlock()

        # 10 B-Block
        self.b_block_1 = BBlock("B_BLOCK_1_")
        self.b_block_2 = BBlock("B_BLOCK_2_")
        self.b_block_3 = BBlock("B_BLOCK_3_")
        self.b_block_4 = BBlock("B_BLOCK_4_")
        self.b_block_5 = BBlock("B_BLOCK_5_")
        self.b_block_6 = BBlock("B_BLOCK_6_")
        self.b_block_7 = BBlock("B_BLOCK_7_")
        self.b_block_8 = BBlock("B_BLOCK_8_")
        self.b_block_9 = BBlock("B_BLOCK_9_")
        self.b_block_10 = BBlock("B_BLOCK_10_")

        # Reduction B-Block
        self.reduction_b_block = ReductionBBlock()

        # 5 C-Block
        self.c_block_1 = CBlock("C_BLOCK_1_")
        self.c_block_2 = CBlock("C_BLOCK_2_")
        self.c_block_3 = CBlock("C_BLOCK_3_")
        self.c_block_4 = CBlock("C_BLOCK_4_")
        self.c_block_5 = CBlock("C_BLOCK_5_")

        # Average block
        self.global_average = GlobalAveragePooling2D(name='AvgPool')
        self.dropout = Dropout(0.2, name='Dropout')

        # Bottleneck
        self.bottle_neck = Dense(128, use_bias=False, name='Bottleneck')
        self.bottle_neck_normalization = BatchNormalization(momentum=0.995, epsilon=0.001,
                                                            scale=False, name='Bottleneck_BatchNorm')

        # Classify
        if number_of_class:
            if arc_face:
                self.classify = ArcFace(number_of_class, )
            else:
                self.classify = Dense(
                    number_of_class, use_bias=False, name='Bottleneck_train')

    def call(self, inputs, training=False):

        x = self.stem(inputs)

        x = self.a_block_1(x)
        x = self.a_block_2(x)
        x = self.a_block_3(x)
        x = self.a_block_4(x)
        x = self.a_block_5(x)
        x = self.reduction_a_block(x)

        x = self.b_block_1(x)
        x = self.b_block_2(x)
        x = self.b_block_3(x)
        x = self.b_block_4(x)
        x = self.b_block_5(x)
        x = self.b_block_6(x)
        x = self.b_block_7(x)
        x = self.b_block_8(x)
        x = self.b_block_9(x)
        x = self.b_block_10(x)
        x = self.reduction_b_block(x)

        x = self.c_block_1(x)
        x = self.c_block_2(x)
        x = self.c_block_3(x)
        x = self.c_block_4(x)
        x = self.c_block_5(x)

        x = self.global_average(x)
        x = self.dropout(x)
        x = self.bottle_neck(x)
        x = self.bottle_neck_normalization(x)

        if self.number_of_class and (not self.arc_face):
            x = self.classify(x)
        if self.number_of_class and self.arc_face:
            x = self.classify((x, label))
        return x

    def train_step(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            logits = self(images, training=True)
            loss = self.compiled_loss(labels, logits)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        # Update metrics
        self.compiled_metrics.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}


def scaling(x, scale):
    return x * scale


class Stem(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__()
        self.stem_conv_1 = Conv2D(
            32, 3, strides=2, padding='valid', use_bias=False, name=name + 'Conv2d_1a_3x3')
        self.stem_batch_1 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Conv2d_1a_3x3_BatchNorm')
        self.stem_activation_1 = LeakyReLU(
            name=name + 'Conv2d_1a_3x3_Activation')

        self.stem_conv_2 = Conv2D(
            32, 3, strides=1, padding='valid', use_bias=False, name=name + 'Conv2d_2a_3x3')
        self.stem_batch_2 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Conv2d_2a_3x3_BatchNorm')
        self.stem_activation_2 = LeakyReLU(
            name=name + 'Conv2d_2a_3x3_Activation')

        self.stem_conv_3 = Conv2D(
            64, 3, strides=1, padding='same', use_bias=False, name=name + 'Conv2d_2b_3x3')
        self.stem_batch_3 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Conv2d_2b_3x3_BatchNorm')
        self.stem_activation_3 = LeakyReLU(
            name=name + 'Conv2d_2b_3x3_Activation')
        self.stem_max_pooling_3 = MaxPooling2D(
            3, strides=2, name=name + 'MaxPool_3a_3x3')

        self.stem_conv_4 = Conv2D(
            80, 1, strides=1, padding='valid', use_bias=False, name=name + 'Conv2d_3b_1x1')
        self.stem_batch_4 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Conv2d_3b_1x1_BatchNorm')
        self.stem_activation_4 = LeakyReLU(
            name=name + 'Conv2d_3b_1x1_Activation')

        self.stem_conv_5 = Conv2D(
            192, 3, strides=1, padding='valid', use_bias=False, name=name + 'Conv2d_4a_3x3')
        self.stem_batch_5 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Conv2d_4a_3x3_BatchNorm')
        self.stem_activation_5 = LeakyReLU(
            name=name + 'Conv2d_4a_3x3_Activation')

        self.stem_conv_6 = Conv2D(
            256, 3, strides=2, padding='valid', use_bias=False, name=name + 'Conv2d_4b_3x3')
        self.stem_batch_6 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Conv2d_4b_3x3_BatchNorm')
        self.stem_activation_6 = LeakyReLU(
            name=name + 'Conv2d_4b_3x3_Activation')

    def call(self, inputs):
        x = self.stem_conv_1(inputs)
        x = self.stem_batch_1(x)
        x = self.stem_activation_1(x)

        x = self.stem_conv_2(x)
        x = self.stem_batch_2(x)
        x = self.stem_activation_2(x)

        x = self.stem_conv_3(x)
        x = self.stem_batch_3(x)
        x = self.stem_activation_3(x)
        x = self.stem_max_pooling_3(x)

        x = self.stem_conv_4(x)
        x = self.stem_batch_4(x)
        x = self.stem_activation_4(x)

        x = self.stem_conv_5(x)
        x = self.stem_batch_5(x)
        x = self.stem_activation_5(x)

        x = self.stem_conv_6(x)
        x = self.stem_batch_6(x)
        x = self.stem_activation_6(x)
        return x


class ABlock(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__()
        self.name_layer = name
        self.branch_0_conv = Conv2D(32, 1, strides=1, padding='same',
                                    use_bias=False, name=name + 'Block35_Branch_0_Conv2d_1x1')
        self.branch_0_batch = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Block35_Branch_0_Conv2d_1x1_BatchNorm')
        self.branch_0_activation = LeakyReLU(
            name=name + 'Block35_Branch_0_Conv2d_1x1_Activation')

        self.branch_1_conv_1 = Conv2D(32, 1, strides=1, padding='same',
                                      use_bias=False, name=name + 'Block35_Branch_1_Conv2d_0a_1x1')
        self.branch_1_batch_1 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Block35_Branch_1_Conv2d_0a_1x1_BatchNorm')
        self.branch_1_activation_1 = LeakyReLU(
            name=name + 'Block35_1_Branch_1_Conv2d_0a_1x1_Activation')
        self.branch_1_conv_2 = Conv2D(32, 3, strides=1, padding='same',
                                      use_bias=False, name=name + 'Block35_Branch_1_Conv2d_0b_3x3')
        self.branch_1_batch_2 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Block35_Branch_1_Conv2d_0b_3x3_BatchNorm')
        self.branch_1_activation_1 = LeakyReLU(
            name=name + 'Block35_Branch_1_Conv2d_0b_3x3_Activation')

        self.branch_2_conv_1 = Conv2D(32, 1, strides=1, padding='same',
                                      use_bias=False, name=name + 'Block35_Branch_2_Conv2d_0a_1x1')
        self.branch_2_batch_1 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Block35_Branch_2_Conv2d_0a_1x1_BatchNorm')
        self.branch_2_activation_1 = LeakyReLU(
            name=name + 'Block35_Branch_2_Conv2d_0a_1x1_Activation')
        self.branch_2_conv_2 = Conv2D(32, 3, strides=1, padding='same',
                                      use_bias=False, name=name + 'Block35_Branch_2_Conv2d_0b_3x3')
        self.branch_2_batch_2 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Block35_Branch_2_Conv2d_0b_3x3_BatchNorm')
        self.branch_2_activation_2 = LeakyReLU(
            name=name + 'Block35_Branch_2_Conv2d_0b_3x3_Activation')
        self.branch_2_conv_3 = Conv2D(32, 3, strides=1, padding='same',
                                      use_bias=False, name=name + 'Block35_Branch_2_Conv2d_0c_3x3')
        self.branch_2_batch_3 = BatchNormalization(
            axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name + 'Block35_Branch_2_Conv2d_0c_3x3_BatchNorm')
        self.branch_2_activation_3 = LeakyReLU(
            name=name + 'Block35_Branch_2_Conv2d_0c_3x3_Activation')
        self.mixed = Concatenate(axis=3, name=name + 'Block35_Concatenate')
        self.format_channel = Conv2D(256, 1, strides=1, padding='same',
                                     use_bias=True, name=name + 'Block35_Conv2d_1x1')
        self.add = Add(name=name + "Addding")
        self.output_activation = LeakyReLU(name=name + 'Block35_Activation')

    def call(self, inputs):
        branch_0 = self.branch_0_conv(inputs)
        branch_0 = self.branch_0_batch(branch_0)
        branch_0 = self.branch_0_activation(branch_0)

        branch_1 = self.branch_1_conv_1(inputs)
        branch_1 = self.branch_1_batch_1(branch_1)
        branch_1 = self.branch_1_activation_1(branch_1)
        branch_1 = self.branch_1_conv_2(branch_1)
        branch_1 = self.branch_1_batch_2(branch_1)
        branch_1 = self.branch_1_activation_1(branch_1)

        branch_2 = self.branch_2_conv_1(inputs)
        branch_2 = self.branch_2_batch_1(branch_2)
        branch_2 = self.branch_2_activation_1(branch_2)
        branch_2 = self.branch_2_conv_2(branch_2)
        branch_2 = self.branch_2_batch_2(branch_2)
        branch_2 = self.branch_2_activation_2(branch_2)
        branch_2 = self.branch_2_conv_3(branch_2)
        branch_2 = self.branch_2_batch_3(branch_2)
        branch_2 = self.branch_2_activation_3(branch_2)

        branches = [branch_0, branch_1, branch_2]
        mixed = self.mixed(branches)
        up = self.format_channel(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[
            1:], arguments={'scale': 0.17}, name=self.name_layer + "Scaling")(up)
        inputs = self.add([inputs, up])
        inputs = self.output_activation(inputs)

        return inputs


class ReductionABlock(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__()
        self.branch_0_conv_1 = Conv2D(384, 3, strides=2, padding='valid',
                                      use_bias=False, name=name + 'Mixed_6a_Branch_0_Conv2d_1a_3x3')
        self.branch_0_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name + 'Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')
        self.branch_0_activation_1 = LeakyReLU(
            name=name + 'Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')

        self.branch_1_conv_1 = Conv2D(192, 1, strides=1, padding='same',
                                      use_bias=False, name=name + 'Mixed_6a_Branch_1_Conv2d_0a_1x1')
        self.branch_1_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name + 'Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')
        self.branch_1_activation_1 = LeakyReLU(
            name=name + 'Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')
        self.branch_1_conv_2 = Conv2D(192, 3, strides=1, padding='same', use_bias=False,
                                      name=name + 'Mixed_6a_Branch_1_Conv2d_0b_3x3')
        self.branch_1_batch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name + 'Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')
        self.branch_1_activation_2 = LeakyReLU(
            name=name + 'Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')
        self.branch_1_conv_3 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                                      name=name + 'Mixed_6a_Branch_1_Conv2d_1a_3x3')
        self.branch_1_batch_3 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name + 'Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')
        self.branch_1_activation_3 = LeakyReLU(
            name=name + 'Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')
        self.branch_pool = MaxPooling2D(
            3, strides=2, padding='valid', name=name + 'Mixed_6a_Branch_2_MaxPool_1a_3x3')

        self.mixed = Concatenate(axis=3, name=name + 'Mixed_6a')

    def call(self, inputs):
        branch_0 = self.branch_0_conv_1(inputs)
        branch_0 = self.branch_0_batch_1(branch_0)
        branch_0 = self.branch_0_activation_1(branch_0)

        branch_1 = self.branch_1_conv_1(inputs)
        branch_1 = self.branch_1_batch_1(branch_1)
        branch_1 = self.branch_1_activation_1(branch_1)

        branch_1 = self.branch_1_conv_2(branch_1)
        branch_1 = self.branch_1_batch_2(branch_1)
        branch_1 = self.branch_1_activation_2(branch_1)

        branch_1 = self.branch_1_conv_3(branch_1)
        branch_1 = self.branch_1_batch_3(branch_1)
        branch_1 = self.branch_1_activation_3(branch_1)

        branch_pool = self.branch_pool(inputs)
        branches = [branch_0, branch_1, branch_pool]
        x = self.mixed(branches)
        return x


class BBlock(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__()
        self.layer_name = name
        self.branch_0_conv_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False,
                                      name=name+'Block17_Branch_0_Conv2d_1x1')
        self.branch_0_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Block17_Branch_0_Conv2d_1x1_BatchNorm')
        self.branch_0_activation_1 = LeakyReLU(
            name=name+'Block17_Branch_0_Conv2d_1x1_Activation')

        self.branch_1_conv_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False,
                                      name=name+'Block17_Branch_1_Conv2d_0a_1x1')
        self.branch_1_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                                   name=name+'Block17_Branch_1_Conv2d_0a_1x1_BatchNorm')
        self.branch_1_activation_1 = LeakyReLU(
            name=name+'Block17_Branch_1_Conv2d_0a_1x1_Activation')

        self.branch_1_conv_2 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                                      name=name+'Block17_Branch_1_Conv2d_0b_1x7')
        self.branch_1_batch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                                   name=name+'Block17_Branch_1_Conv2d_0b_1x7_BatchNorm')
        self.branch_1_activation_2 = LeakyReLU(
            name=name+'Block17_Branch_1_Conv2d_0b_1x7_Activation')

        self.branch_1_conv_3 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                                      name=name+'Block17_Branch_1_Conv2d_0c_7x1')
        self.branch_1_batch_3 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                                   name=name+'Block17_Branch_1_Conv2d_0c_7x1_BatchNorm')
        self.branch_1_activation_3 = LeakyReLU(
            name=name+'Block17_Branch_1_Conv2d_0c_7x1_Activation')

        self.mixed = Concatenate(axis=3, name=name+'Block17_Concatenate')
        self.format_channel = Conv2D(896, 1, strides=1, padding='same', use_bias=True,
                                     name=name+'Block17_Conv2d_1x1')

        self.add = Add(name=name + "Addding")
        self.output_activation = LeakyReLU(name=name+'Block17_Activation')

    def call(self, inputs):
        branch_0 = self.branch_0_conv_1(inputs)
        branch_0 = self.branch_0_batch_1(branch_0)
        branch_0 = self.branch_0_activation_1(branch_0)

        branch_1 = self.branch_1_conv_1(inputs)
        branch_1 = self.branch_1_batch_1(branch_1)
        branch_1 = self.branch_1_activation_1(branch_1)

        branch_1 = self.branch_1_conv_2(branch_1)
        branch_1 = self.branch_1_batch_2(branch_1)
        branch_1 = self.branch_1_activation_2(branch_1)

        branch_1 = self.branch_1_conv_3(branch_1)
        branch_1 = self.branch_1_batch_3(branch_1)
        branch_1 = self.branch_1_activation_3(branch_1)

        branches = [branch_0, branch_1]
        mixed = self.mixed(branches)
        up = self.format_channel(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={
                    'scale': 0.1}, name=self.layer_name + "Scaling")(up)
        inputs = self.add([inputs, up])
        inputs = self.output_activation(inputs)
        return inputs


class ReductionBBlock(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__()
        self.layer_name = name
        self.branch_0_conv_1 = Conv2D(256, 1, strides=1, padding='same',
                                      use_bias=False, name=name+'Mixed_7a_Branch_0_Conv2d_0a_1x1')
        self.branch_0_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')
        self.branch_0_activation_1 = LeakyReLU(
            name=name+'Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')
        self.branch_0_conv_2 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False,
                                      name=name+'Mixed_7a_Branch_0_Conv2d_1a_3x3')
        self.branch_0_batch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')
        self.branch_0_activation_2 = LeakyReLU(
            name=name+'Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')

        self.branch_1_conv_1 = Conv2D(256, 1, strides=1, padding='same',
                                      use_bias=False, name=name+'Mixed_7a_Branch_1_Conv2d_0a_1x1')
        self.branch_1_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')
        self.branch_1_activation_1 = LeakyReLU(
            name=name+'Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')
        self.branch_1_conv_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                                      name=name+'Mixed_7a_Branch_1_Conv2d_1a_3x3')
        self.branch_1_batch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')
        self.branch_1_activation_2 = LeakyReLU(
            name=name+'Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')

        self.branch_2_conv_1 = Conv2D(256, 1, strides=1, padding='same',
                                      use_bias=False, name=name+'Mixed_7a_Branch_2_Conv2d_0a_1x1')
        self.branch_2_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')
        self.branch_2_activation_1 = LeakyReLU(
            name=name+'Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')
        self.branch_2_conv_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False,
                                      name=name+'Mixed_7a_Branch_2_Conv2d_0b_3x3')
        self.branch_2_batch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')
        self.branch_2_activation_2 = LeakyReLU(
            name=name+'Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')
        self.branch_2_conv_3 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False,
                                      name=name+'Mixed_7a_Branch_2_Conv2d_1a_3x3')
        self.branch_2_batch_3 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')
        self.branch_2_activation_3 = LeakyReLU(
            name=name+'Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')
        self.branch_pool = MaxPooling2D(
            3, strides=2, padding='valid', name=name+'Mixed_7a_Branch_3_MaxPool_1a_3x3')
        self.mixed = Concatenate(axis=3, name=name+'Mixed_7a')

    def call(self, inputs):
        branch_0 = self.branch_0_conv_1(inputs)
        branch_0 = self.branch_0_batch_1(branch_0)
        branch_0 = self.branch_0_activation_1(branch_0)
        branch_0 = self.branch_0_conv_2(branch_0)
        branch_0 = self.branch_0_batch_2(branch_0)
        branch_0 = self.branch_0_activation_2(branch_0)

        branch_1 = self.branch_1_conv_1(inputs)
        branch_1 = self.branch_1_batch_1(branch_1)
        branch_1 = self.branch_1_activation_1(branch_1)
        branch_1 = self.branch_1_conv_2(branch_1)
        branch_1 = self.branch_1_batch_2(branch_1)
        branch_1 = self.branch_1_activation_2(branch_1)

        branch_2 = self.branch_2_conv_1(inputs)
        branch_2 = self.branch_2_batch_1(branch_2)
        branch_2 = self.branch_2_activation_1(branch_2)
        branch_2 = self.branch_2_conv_2(branch_2)
        branch_2 = self.branch_2_batch_2(branch_2)
        branch_2 = self.branch_2_activation_2(branch_2)
        branch_2 = self.branch_2_conv_3(branch_2)
        branch_2 = self.branch_2_batch_3(branch_2)
        branch_2 = self.branch_2_activation_3(branch_2)

        branch_pool = self.branch_pool(inputs)
        branches = [branch_0, branch_1, branch_2, branch_pool]

        output = self.mixed(branches)
        return output


class CBlock(tf.keras.layers.Layer):
    def __init__(self, name=""):
        super().__init__()
        self.layer_name = name
        self.branch_0_conv_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False,
                                      name=name+'Block8_1_Branch_0_Conv2d_1x1')
        self.branch_0_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001,
                                                   scale=False, name=name+'Block8_1_Branch_0_Conv2d_1x1_BatchNorm')
        self.branch_0_activation_1 = LeakyReLU(
            name=name+'Block8_1_Branch_0_Conv2d_1x1_Activation')

        self.branch_1_conv_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False,
                                      name=name+'Block8_1_Branch_1_Conv2d_0a_1x1')
        self.branch_1_batch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                                   name=name+'Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')
        self.branch_1_activation_1 = LeakyReLU(
            name=name+'Block8_1_Branch_1_Conv2d_0a_1x1_Activation')
        self.branch_1_conv_2 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                                      name=name+'Block8_1_Branch_1_Conv2d_0b_1x3')
        self.branch_1_batch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                                   name=name+'Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')
        self.branch_1_activation_2 = LeakyReLU(
            name=name+'Block8_1_Branch_1_Conv2d_0b_1x3_Activation')
        self.branch_1_conv_3 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                                      name=name+'Block8_1_Branch_1_Conv2d_0c_3x1')
        self.branch_1_batch_3 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                                   name=name+'Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')
        self.branch_1_activation_3 = LeakyReLU(
            name=name+'Block8_1_Branch_1_Conv2d_0c_3x1_Activation')
        self.mixed = Concatenate(axis=3, name=name+'Block8_1_Concatenate')
        self.format_channel = Conv2D(1792, 1, strides=1, padding='same', use_bias=True,
                                     name=name+'Block8_1_Conv2d_1x1')

        self.add = Add(name=name + "Addding")
        self.output_activation = LeakyReLU(name=name+'Block8_1_Activation')

    def call(self, inputs):
        branch_0 = self.branch_0_conv_1(inputs)
        branch_0 = self.branch_0_batch_1(branch_0)
        branch_0 = self.branch_0_activation_1(branch_0)

        branch_1 = self.branch_1_conv_1(inputs)
        branch_1 = self.branch_1_batch_1(branch_1)
        branch_1 = self.branch_1_activation_1(branch_1)
        branch_1 = self.branch_1_conv_2(branch_1)
        branch_1 = self.branch_1_batch_2(branch_1)
        branch_1 = self.branch_1_activation_2(branch_1)
        branch_1 = self.branch_1_conv_3(branch_1)
        branch_1 = self.branch_1_batch_3(branch_1)
        branch_1 = self.branch_1_activation_3(branch_1)

        branches = [branch_0, branch_1]
        mixed = self.mixed(branches)
        up = self.format_channel(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={
                    'scale': 0.2}, name=self.layer_name + "Scaling")(up)
        output = self.add([inputs, up])
        output = self.output_activation(output)
        return output


def _resolve_training_flag(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return training


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



class ArcFace(tf.keras.layers.Layer):
    """
    Implementation of ArcFace layer. Reference: https://arxiv.org/abs/1801.07698

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
                 regularizer=regularizers.l2(),
                 name='arcface',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self._s = float(s)
        self._m = float(m)
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self.num_classes),
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

        training = _resolve_training_flag(self, training)
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

def call_instance_model(input_shape, number_of_class, embedding, model_name = "Inception-Resnet-V1",\
                        last_layer = "Dense"):
    
    # Create model structure
    embedding_model = None
    if model_name == "Inception-Resnet-V1":
        embedding_model = InceptionResNetV1(input_shape, embedding)
    elif model_name == "Inception-Resnet-V2":
        embedding_model = tf.keras.applications.InceptionResNetV2(include_top=True, weights=None,\
                        input_shape=input_shape, classes = embedding)
        
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
