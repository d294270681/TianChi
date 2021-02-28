train_data_path = './data/train.csv'
test_data_path = './data/test.csv'

pretrained_model = "bert"

ckpt_path = './model/ckpt/'+pretrained_model+'/'
log_path = './log/'+pretrained_model+'/'

proccessing_model_path = 'https://tfhub.dev/tensorflow/bert_zh_preprocess/3'
pretrained_model_name = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3'

LABEL_COLUMN = 'label'
batch_size = 64
epochs = 5
seq_length = 128
init_lr = 3e-5
