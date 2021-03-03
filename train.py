import pandas as pd
import numpy as np
from config.train_config import *
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
from official.nlp import optimization
from transformers import BertTokenizer
import os

from model.model import *

tf.get_logger().setLevel('ERROR')


def build_model(loss, optimizer, metrics, model_name):
  if model_name == "bert":
    return get_bert_model(loss, optimizer, metrics)
  if model_name == "roberta":
    return get_roberta_model(loss, optimizer, metrics)

def get_roberta_model(loss, optimizer, metrics):
  model = Roberta_Simmiliar(pretrained_model_path)
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model

def get_bert_model(loss, optimizer, metrics):
  text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input_'+str(i)) for i in range(2)]
  preprocessor = hub.load(proccessing_model_path)
  tokenize = hub.KerasLayer(preprocessor.tokenize, name='BERT_tokenizer')
  tokenized_inputs = [tokenize(segment) for segment in text_inputs]
  bert_pack_inputs = hub.KerasLayer(
      preprocessor.bert_pack_inputs,
      arguments=dict(seq_length=seq_length), name='pack') 
  encoder_inputs = bert_pack_inputs(tokenized_inputs)
  encoder = hub.KerasLayer(pretrained_model_path, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
  model = Bert_Simmilar(text_inputs, net)
  tf.keras.utils.plot_model(model, to_file=ckpt_path+"../model_"+pretrained_model+".png")
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model

def get_callback_method(model_name):
  if model_name == "bert":
    return [tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path+"/match_model_{epoch}:val_loss-{val_loss:.2f}:val_AUC-{val_AUC:.3f}:val_BinAcc-{val_BinaryAccuracy:.3f}",
            save_best_only=True, 
            monitor="val_AUC",
            mode='max',
            verbose=1
        ),tf.keras.callbacks.CSVLogger(log_path+'training.log')]
  if model_name == "roberta":
    return [tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path+"/match_model_{epoch}:val_loss-{val_loss:.2f}:val_AUC-{val_AUC:.3f}:val_BinAcc-{val_BinaryAccuracy:.3f}",
            verbose=1,
            save_weights_only=True
        ),tf.keras.callbacks.CSVLogger(log_path+'training.log')]

def make_or_restore_model(dataset, model_name):
  checkpoints = [ckpt_path + "/" + name for name in os.listdir(ckpt_path)]
  steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)
  optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  metricsA = tf.metrics.BinaryAccuracy(name='BinaryAccuracy')
  metricsB = tf.keras.metrics.AUC(name='AUC')
  if checkpoints:
    latest_checkpoint = max(checkpoints)[0:-6]
    print("Restoring from", latest_checkpoint)
    if model_name == "bert":
      model = tf.keras.models.load_model(latest_checkpoint, compile=False)
    if model_name == "roberta":
      model = Roberta_Simmiliar(pretrained_model_path)
      model.load_weights(latest_checkpoint)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metricsA,metricsB])
    return model
  print("Creating a new model")
  return build_model(loss, optimizer, [metricsA,metricsB], model_name)

def anti_data_to_tensor(tokenizer, data):
  tokenizer.save_vocabulary(ckpt_path+"/../roberta_vocab.txt")
  count = len(open(ckpt_path+"/../roberta_vocab.txt", 'r').readlines())
  data_number = []
  for i in data:
    cls_of_row_1 = [int(k) if int(k)<count else 0 for k in i[0].split(" ")]
    cls_of_row_2 = [int(k) if int(k)<count else 0 for k in i[1].split(" ")]
    if len(cls_of_row_1)+len(cls_of_row_2)+3 <= seq_length:
      row = tokenizer.build_inputs_with_special_tokens(cls_of_row_1, cls_of_row_2)
      row += [0]*(seq_length-len(row))
    else:  
      cls_of_row_2 = cls_of_row_2[0:len(cls_of_row_2)+1-(len(cls_of_row_1)+len(cls_of_row_2)+4-seq_length)]
      row = tokenizer.build_inputs_with_special_tokens(cls_of_row_1, cls_of_row_2)
    data_number.append(row)
  return tf.constant(np.array(data_number))


def train():
  print("create dataset")

  
  if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
  if not os.path.exists(log_path):
    os.makedirs(log_path)

  if pretrained_model=="bert":
    print("choose bert")
    train_data = pd.read_csv(train_data_path)
    train_target = train_data.pop('label')
    train_seq_1, train_seq_2 = np.split(train_data.values,2,1)

    test_data = pd.read_csv(test_data_path)
    test_target = test_data.pop('label')
    test_seq_1, test_seq_2 = np.split(test_data.values,2,1)
  
  if pretrained_model=="roberta":
    print("choose robert")
    tokenizer_roberta = BertTokenizer.from_pretrained(proccessing_model_path)

    train_data = pd.read_csv(train_data_path)
    train_target = train_data.pop('label')
    train_seq_1, train_seq_2 = np.split(train_data.values,2,1)

    test_data = pd.read_csv(test_data_path)
    test_target = test_data.pop('label')
    if not anti:
      print("非脱敏数据处理中")
      train_data = tokenizer_roberta(train_data.values.tolist(),padding='max_length',max_length =seq_length,return_tensors="tf",truncation=True)
      test_data = tokenizer_roberta(test_data.values.tolist(),padding='max_length',max_length =seq_length,return_tensors="tf",truncation=True)
    else:
      print("脱敏数据处理中")
      anti_train_data = tokenizer_roberta(train_data.values.tolist(),padding='max_length',max_length =seq_length,return_tensors="tf",truncation=True)
      anti_test_data = tokenizer_roberta(test_data.values.tolist(),padding='max_length',max_length =seq_length,return_tensors="tf",truncation=True)


      anti_train_data['input_ids'] = anti_data_to_tensor(tokenizer_roberta, train_data.values)
      anti_test_data['input_ids'] = anti_data_to_tensor(tokenizer_roberta, test_data.values)

  train_dataset = tf.data.Dataset.from_tensor_slices({"text_input_0":train_seq_1, "text_input_1":train_seq_2, "label":train_target.values})
  train_dataset = train_dataset.shuffle(len(train_data)).batch(batch_size)


  model = make_or_restore_model(train_dataset, pretrained_model)
 
  callback = get_callback_method(pretrained_model)
  
  if pretrained_model=="bert":
    history = model.fit(x=(train_seq_1, train_seq_2), y = train_target.values, 
      validation_data=((test_seq_1, test_seq_2), test_target.values),
      epochs=epochs, callbacks = callback, batch_size = batch_size)

  if pretrained_model=="roberta":
    if not anti:
      history = model.fit(x = list(train_data.values()), y = train_target.values, 
            validation_data=(list(test_data.values()), test_target.values)
            ,epochs=epochs, callbacks = callback, batch_size = batch_size)
    else:
      history = model.fit(x = list(anti_train_data.values()), y = train_target.values, 
            validation_data=(list(anti_test_data.values()), test_target.values)
            ,epochs=epochs, callbacks = callback, batch_size = batch_size)
  
if __name__ == '__main__':
  print("start training")
  train()