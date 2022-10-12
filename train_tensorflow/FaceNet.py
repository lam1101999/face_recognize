
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.losses import metric_learning
from tensorflow.keras.layers import Input,BatchNormalization,Dropout,Flatten,Dense, Layer
from typing import  Union, Callable
from train_tensorflow.Net import InceptionResNetV1
from train_tensorflow.Net import ArcFace
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


class FaceNetModel(tf.keras.Model):
  def __init__(self, embedding_model, arc_face = False, margin=1):
    super().__init__()
    self.embedding_model

  
  def call(self, inputs):
        return self.network(inputs)

  def train_step(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            logits = self.network(images, training = True)
            loss = self.compiled_loss(labels, logits)

        gradients = tape.gradient(loss, self.network.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
  def test_step(self, inputs):
      images, labels = inputs
      logits = self.network(images, training = False)
      loss = self._compute_loss(labels, logits)

      # Let's update and return the loss metric.
      self.loss_tracker.update_state(loss)
      return {"loss": self.loss_tracker.result()}


def call_instance_FaceNet_with_last_isDense(input_shape, number_of_class):
    embedding_model = InceptionResNetV1(input_shape)
    
    arc_face_model = 
    
    # The face-net model
    outputs = tf.keras.layers.Dense(number_of_class, use_bias=False, name='Bottleneck_train')(embedding_model.output)
    face_net_model = tf.keras.Model(embedding_model.input, outputs, name="FaceNetModel")
    return face_net_model

def call_instance_FaceNet_ArcFace(input_shape, number_of_class):
    embedding_model = InceptionResNetV1(input_shape)
    shape_logit = Input(shape = (number_of_class,))
    
    #ArcFace
    outputs = ArcFace(number_of_class, regularizer = regularizers.l2(1e-4))(tf.stack(embedding_model.output, shape_logit))
    face_net_arc_face_model = tf.keras.Model(tf.stack(embedding_model.input,shape_logit), outputs, name = "FaceNetArcFace")
    return face_net_arc_face_model

def convert_train_model_to_embedding(train_model):
    cut_the_last_layer = tf.keras.models.Model(inputs=train_model.input, outputs=train_model.layers[-2].output)
    outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(cut_the_last_layer.output)
    face_net_model = tf.keras.Model(cut_the_last_layer.input, outputs, name="FaceNetModel")
    return face_net_model


if __name__ == "__main__":
    weight_decay = 1e-4
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))
    a = call_instance_FaceNet_ArcFace([110,110,3],200)