import tensorflow as tf
from transformers import TFBertModel
from config import *

class Bert_Simmilar(tf.keras.Model):
  def train_step(self, data):   
    (seq1 ,seq2) ,y = data
    text_inputs = [seq1, seq2]

    with tf.GradientTape() as tape:
      y_pred = self(text_inputs, training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) 
    
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars) 

    self.optimizer.apply_gradients(zip(gradients, trainable_vars))  
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}
  
  def test_step(self, data):
    (seq1 ,seq2) ,y = data
    text_inputs = [seq1, seq2]

    y_pred = self(text_inputs, training=False)
    self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

class Roberta_Simmiliar(tf.keras.Model):
  def __init__(self, pretrained_model_path, name="Roberta", **kwargs): 
    super(Roberta_Simmiliar, self).__init__(**kwargs)
    self.model = TFBertModel.from_pretrained(pretrained_model_path)
    self.dropout = tf.keras.layers.Dropout(0.1)
    self.dense_layer_1 = tf.keras.layers.Dense(256, activation='relu', name='classifier')
    self.dense_layer_2 = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')
  
  def call(self, inputs):
    encoder_outputs = self.model(**inputs)
    output = encoder_outputs['pooler_output']
    output = self.dropout(output)
    output = self.dense_layer_1(output)
    output = self.dense_layer_2(output)
    return output

  def train_step(self, data):  
    x ,y = data
    input_ids, token_type_ids, attention_mask = x
    x = {
      'input_ids':input_ids,
      'token_type_ids':token_type_ids,
      'attention_mask':attention_mask
    }
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) 
    
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars) 

    self.optimizer.apply_gradients(zip(gradients, trainable_vars))  
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    x ,y = data
    input_ids, token_type_ids, attention_mask = x
    x = {
      'input_ids':input_ids,
      'token_type_ids':token_type_ids,
      'attention_mask':attention_mask
    }
    y_pred = self(x, training=False)
    self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}