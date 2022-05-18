import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, Lambda, add, GlobalAveragePooling2D, Convolution2D, LocallyConnected2D, ZeroPadding2D, concatenate, AveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

def scaling(x, scale):
	return x * scale

def InceptionResNetV1(input_shape = [128,128,3]):
	
	inputs = Input(shape=input_shape)
	x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_1a_3x3') (inputs)
	x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)
	x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)
	x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_2a_3x3') (x)
	x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)
	x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)
	x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name= 'Conv2d_2b_3x3') (x)
	x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)
	x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)
	x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
	x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False, name= 'Conv2d_3b_1x1') (x)
	x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)
	x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)
	x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False, name= 'Conv2d_4a_3x3') (x)
	x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)
	x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)
	x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Conv2d_4b_3x3') (x)
	x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)
	x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)
	
	# 5x Block35 (Inception-ResNet-A block):
	x = A_Block(x, "A_BLOCK_1_")
	x = A_Block(x, "A_BLOCK_2_")
	x = A_Block(x, "A_BLOCK_3_")
	x = A_Block(x, "A_BLOCK_4_")
	x = A_Block(x, "A_BLOCK_5_")

	# Mixed 6a (Reduction-A block):
	branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_0_Conv2d_1a_3x3') (x)
	branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
	branch_0 = Activation('relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
	branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0a_1x1') (x)
	branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
	branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
	branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_0b_3x3') (branch_1)
	branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
	branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
	branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_6a_Branch_1_Conv2d_1a_3x3') (branch_1)
	branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
	branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
	branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
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
	branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_0a_1x1') (x)
	branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
	branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
	branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_0_Conv2d_1a_3x3') (branch_0)
	branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
	branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
	branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_0a_1x1') (x)
	branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
	branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
	branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_1_Conv2d_1a_3x3') (branch_1)
	branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
	branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
	branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0a_1x1') (x)
	branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
	branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
	branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_0b_3x3') (branch_2)
	branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
	branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
	branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name= 'Mixed_7a_Branch_2_Conv2d_1a_3x3') (branch_2)
	branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
	branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
	branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
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
	x = Dropout(0.2, name='Dropout')(x)
	# Bottleneck
	x = Dense(128, use_bias=False, name='Bottleneck')(x)
	x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)
	# Create model
	model = Model(inputs, x, name='inception_resnet_v2')

	return model

def A_Block(inputs, name = "A_BLOCK"):
	with tf.name_scope(name):
		branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= name +'Block35_Branch_0_Conv2d_1x1') (inputs)
		branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name +'Block35_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
		branch_0 = Activation('relu', name=name +'Block35_Branch_0_Conv2d_1x1_Activation')(branch_0)
		branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= name +'Block35_Branch_1_Conv2d_0a_1x1') (inputs)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name= name +'Block35_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name= name +'Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
		branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= name +'Block35_Branch_1_Conv2d_0b_3x3') (branch_1)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name +'Block35_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name= name +'Block35_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
		branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name= name +'Block35_Branch_2_Conv2d_0a_1x1') (inputs)
		branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name= name +'Block35_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
		branch_2 = Activation('relu', name= name +'Block35_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
		branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= name +'Block35_Branch_2_Conv2d_0b_3x3') (branch_2)
		branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name= name +'Block35_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
		branch_2 = Activation('relu', name= name +'Block35_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
		branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name= name +'Block35_Branch_2_Conv2d_0c_3x3') (branch_2)
		branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name= name +'Block35_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
		branch_2 = Activation('relu', name= name +'Block35_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
		branches = [branch_0, branch_1, branch_2]
		mixed = Concatenate(axis=3, name= name +'Block35_Concatenate')(branches)
		up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name= name +'Block35_Conv2d_1x1') (mixed)
		up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17}, name = name +"Scaling")(up)
		inputs = add([inputs, up], name = name +"Addding")
		inputs = Activation('relu', name= name +'Block35_Activation')(inputs)
	return inputs

def B_Block(inputs, name = "B_BLOCK"):
	with tf.name_scope(name):
		branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= name+'Block17_Branch_0_Conv2d_1x1') (inputs)
		branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block17_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
		branch_0 = Activation('relu', name=name+'Block17_Branch_0_Conv2d_1x1_Activation')(branch_0)
		branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name= name+'Block17_Branch_1_Conv2d_0a_1x1') (inputs)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block17_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name=name+'Block17_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
		branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False, name= name+'Block17_Branch_1_Conv2d_0b_1x7') (branch_1)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block17_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name=name+'Block17_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
		branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False, name= name+'Block17_Branch_1_Conv2d_0c_7x1') (branch_1)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block17_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name=name+'Block17_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
		branches = [branch_0, branch_1]
		mixed = Concatenate(axis=3, name=name+'Block17_Concatenate')(branches)
		up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name= name+'Block17_Conv2d_1x1') (mixed)
		up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1}, name = name +"Scaling")(up)
		inputs = add([inputs, up], name = name +"Addding")
		inputs = Activation('relu', name=name+'Block17_Activation')(inputs)
	return inputs

def C_Block(inputs, name = "C_BLOCK"):
	with tf.name_scope(name):
		branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= name+'Block8_1_Branch_0_Conv2d_1x1') (inputs)
		branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
		branch_0 = Activation('relu', name=name+'Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
		branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name= name+'Block8_1_Branch_1_Conv2d_0a_1x1') (inputs)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name=name+'Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
		branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False, name= name+'Block8_1_Branch_1_Conv2d_0b_1x3') (branch_1)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name=name+'Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
		branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False, name= name+'Block8_1_Branch_1_Conv2d_0c_3x1') (branch_1)
		branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=name+'Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
		branch_1 = Activation('relu', name=name+'Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
		branches = [branch_0, branch_1]
		mixed = Concatenate(axis=3, name=name+'Block8_1_Concatenate')(branches)
		up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name= name+'Block8_1_Conv2d_1x1') (mixed)
		up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2}, name = name +"Scaling")(up)
		inputs = add([inputs, up], name = name +"Addding")
		inputs = Activation('relu', name=name+'Block8_1_Activation')(inputs)
	return inputs

if __name__ == "__main__":
	a = InceptionResNetV1([110,110,3])
	a.summary()