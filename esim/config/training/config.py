train_data_path = '../data/data_A/train.csv'
dev_data_path = '../data/data_A/test.csv'
test_data_path = '../data/data_A/testA.csv'

train_file = "./data/train_data.pkl"
valid_file = "./data/dev_data.pkl"

target_dir = "./checkpoint"

proccessed_data = './data'

labeldict =  {"0": 0,"1": 1}
bos = "_BOS_"
eos = "_EOS_"
batch_size = 64
patience = 5
max_gradient_norm = 10.0
lr = 0.0004
checkpoint = None
epochs = 64


vocab_size = 20604
