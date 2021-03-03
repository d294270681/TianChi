from model.model import *
from config.train_config import *
import numpy as np
import os
from train import *
import pandas as pd

def evaluate(model_name, latest_checkpoint):
  if model_name == "roberta":
    test_data = pd.read_csv(test_data_path)
    test_target = test_data.pop('label')

    train_data = pd.read_csv(train_data_path)
    train_target = train_data.pop('label')
    train_seq_1, train_seq_2 = np.split(train_data.values,2,1)

    train_dataset = tf.data.Dataset.from_tensor_slices({"text_input_0":train_seq_1, "text_input_1":train_seq_2, "label":train_target.values})
    train_dataset = train_dataset.shuffle(len(train_data)).batch(batch_size)
    tokenizer_roberta = BertTokenizer.from_pretrained(proccessing_model_path)
    
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metricsA = tf.metrics.BinaryAccuracy(name='BinaryAccuracy')
    metricsB = tf.keras.metrics.AUC(name='AUC')
    model = Roberta_Simmiliar(pretrained_model_path)
    model.load_weights(latest_checkpoint)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metricsA,metricsB])

    anti_test_data = tokenizer_roberta(test_data.values.tolist(),padding='max_length',max_length =seq_length,return_tensors="tf",truncation=True)

    anti_test_data['input_ids'] = anti_data_to_tensor(tokenizer_roberta, test_data.values)

    history = model.evaluate(x = list(anti_test_data.values()), y = test_target.values)

if __name__ == '__main__':
  print("start evaluating")
  evaluate(pretrained_model, "/content/drive/MyDrive/TianChi/ckpt/roberta/match_model_5:val_loss-0.43:val_AUC-0.899:val_BinAcc-0.822")
