#train_data_path = './data/train.csv' 
#test_data_path = './data/test.csv'
train_data_path = './data/data_A/train.csv' 
test_data_path = './data/data_A/test.csv'

testA_data_path = './data/data_A/testA.csv'

anti = True 
#anti = False

#pretrained_model = "bert" 
pretrained_model = "roberta" 

ckpt_path = './ckpt/'+pretrained_model
log_path = './log/'+pretrained_model

#proccessing_model_path = 'https://tfhub.dev/tensorflow/bert_zh_preprocess/3' 
#pretrained_model_path = 'https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/3' 
proccessing_model_path = "hfl/chinese-roberta-wwm-ext"
pretrained_model_path = "hfl/chinese-roberta-wwm-ext"

result_path = "./result/result.txt"

LABEL_COLUMN = 'label'
batch_size = 64
epochs = 5
seq_length = 128
init_lr = 3e-5
