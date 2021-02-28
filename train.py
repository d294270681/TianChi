import pandas as pd
import numpy as np
from config import *
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
from official.nlp import optimization
import os

tf.get_logger().setLevel('ERROR')

class Bert_Simmilar(tf.keras.Model):
  def train_step(self, data):   
    seq1 = data['seq1']
    seq2 = data['seq2'] 
    y = data['label']

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
    seq1 = data['seq1']
    seq2 = data['seq2'] 
    y = data['label']

    text_inputs = [seq1, seq2]

    y_pred = self(text_inputs, training=False)
    self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    self.compiled_metrics.update_state(y, y_pred)
    return {m.name: m.result() for m in self.metrics}

def build_model(loss, optimizer, metrics):
  text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input_'+str(i)) for i in range(2)]
  preprocessor = hub.load(proccessing_model_path)
  tokenize = hub.KerasLayer(preprocessor.tokenize, name='BERT_tokenizer')
  tokenized_inputs = [tokenize(segment) for segment in text_inputs]
  bert_pack_inputs = hub.KerasLayer(
      preprocessor.bert_pack_inputs,
      arguments=dict(seq_length=seq_length), name='pack') 
  encoder_inputs = bert_pack_inputs(tokenized_inputs)
  encoder = hub.KerasLayer(pretrained_model_name, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
  model = Bert_Simmilar(text_inputs, net)
  tf.keras.utils.plot_model(model, to_file=ckpt_path+"../model_"+pretrained_model+".png")
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model

def get_callback_method():
  return [tf.keras.callbacks.ModelCheckpoint(
          filepath=ckpt_path+"match_model_{epoch}:val_loss-{val_loss:.2f}:val_AUC-{val_AUC:.3f}:val_BinAcc-{val_BinaryAccuracy:.3f}",
          save_best_only=True, 
          monitor="val_AUC",
          verbose=1,
      ),tf.keras.callbacks.CSVLogger(log_path+'training.log')]

def make_or_restore_model(dataset):
  checkpoints = [ckpt_path + "/" + name for name in os.listdir(ckpt_path)]
  steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)
  optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metricsA = tf.metrics.BinaryAccuracy(name='BinaryAccuracy')
  metricsB = tf.keras.metrics.AUC(name='AUC')
  if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print("Restoring from", latest_checkpoint)
    model = tf.keras.models.load_model(latest_checkpoint, compile=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metricsA,metricsB])
    return model
  print("Creating a new model")
  return build_model(loss, optimizer, [metricsA,metricsB])

def train():
  print("create dataset")
  train_data = pd.read_csv(train_data_path)
  train_target = train_data.pop('label')
  train_seq_1, train_seq_2 = np.split(train_data.values,2,1)
  train_dataset = tf.data.Dataset.from_tensor_slices({"seq1":train_seq_1, "seq2":train_seq_2, "label":train_target.values})
  test_data = pd.read_csv(test_data_path)
  test_target = test_data.pop('label')
  test_seq_1, test_seq_2 = np.split(test_data.values,2,1)
  test_dataset = tf.data.Dataset.from_tensor_slices({"seq1":test_seq_1, "seq2":test_seq_2, "label":test_target.values})

  train_dataset = train_dataset.shuffle(len(train_data)).batch(batch_size)
  test_dataset = test_dataset.shuffle(len(test_data)).batch(batch_size)

  if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  model = make_or_restore_model(train_dataset)
 
  callback = get_callback_method()
  history = model.fit(x=train_dataset, validation_data=test_dataset, epochs=epochs, callbacks = callback)
  
if __name__ == '__main__':
  print("start training")
  train()