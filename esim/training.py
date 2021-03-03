import pandas as pd
from esim.data import NLIDataset
from collections import Counter
from esim.data import Preprocessor
from esim.data import NLIDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from esim.model import ESIM
from utils import validate, train
from config.training.config import *
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def create_preccess_data(data):
  data = data.rename(columns={"seq_1":"premises", "seq_2":"hypotheses", "label":"labels"})
  data['ids'] = data.index
  premises = data["premises"].values.tolist()
  premises = [i.split() for i in premises]
  hypotheses = data["hypotheses"].values.tolist()
  hypotheses = [i.split() for i in hypotheses]
  labels = data["labels"].values.tolist()
  ids = data["ids"].values.tolist()

  data = {"ids": ids,"premises": premises,"hypotheses": hypotheses,"labels": labels}
  return data

def create_training_data(preprocessor, data, shuffle):
  data = preprocessor.transform_to_indices(data)
  data = NLIDataset(data)
  data = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
  return data

def preproccess_data(train_data_path, dev_data_path, test_data_path):
  train_data = pd.read_csv(train_data_path)
  dev_data = pd.read_csv(dev_data_path)
  test_data = pd.read_csv(test_data_path)
  data = pd.concat((train_data, dev_data, test_data))

  data = create_preccess_data(data)
  train_data = create_preccess_data(train_data)
  dev_data = create_preccess_data(dev_data)
  test_data = create_preccess_data(test_data)
    
  preprocessor = Preprocessor(labeldict=labeldict, bos=bos, eos=eos)
  preprocessor.build_worddict(data)

  train_data = preprocessor.transform_to_indices(train_data)
  dev_data = preprocessor.transform_to_indices(dev_data)
  test_data = preprocessor.transform_to_indices(test_data)

  print("\t* Saving train_data...")
  with open(os.path.join(proccessed_data, "train_data.pkl"), "wb") as pkl_file:
    pickle.dump(train_data, pkl_file)
  print("\t* Saving dev_data...")
  with open(os.path.join(proccessed_data, "dev_data.pkl"), "wb") as pkl_file:
    pickle.dump(dev_data, pkl_file)
  print("\t* Saving test_data...")
  with open(os.path.join(proccessed_data, "test_data.pkl"), "wb") as pkl_file:
    pickle.dump(test_data, pkl_file)

def training(checkpoint):
  if not os.listdir(proccessed_data):
    print("preproccessing data")
    preproccess_data(train_data_path, dev_data_path, test_data_path)

  print("\t* Loading training data...")
  with open(train_file, "rb") as pkl:
    train_data = NLIDataset(pickle.load(pkl))

  print("\t* Loading val data...")
  with open(valid_file, "rb") as pkl:
    dev_data = NLIDataset(pickle.load(pkl))

  train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
  dev_loader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
  model = ESIM(vocab_size,
      300,
      300,
      dropout=0.5,
      num_classes=2,
      device = device).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=0.5,patience=0)
  best_score = 0.0
  start_epoch = 1
  epochs_count = []
  train_losses = []
  valid_losses = []

  if checkpoint:
      checkpoint = torch.load(checkpoint)
      start_epoch = checkpoint["epoch"] + 1
      best_score = checkpoint["best_score"]

      print("\t* Training will continue on existing model from epoch {}..."
                .format(start_epoch))

      model.load_state_dict(checkpoint["model"])
      optimizer.load_state_dict(checkpoint["optimizer"])
      epochs_count = checkpoint["epochs_count"]
      train_losses = checkpoint["train_losses"]
      valid_losses = checkpoint["valid_losses"]

  print(len(dev_loader))
  print(len(dev_loader.dataset))
  _, valid_loss, valid_accuracy, valid_AUC = validate(model,
                        dev_loader,
                        criterion)
  print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, AUC: {:.4f}\n"
    .format(valid_loss, (valid_accuracy*100), valid_AUC))

  print("\n",20 * "=","Training ESIM model on device: {}".format(device),20 * "=")

  patience_counter = 0
  for epoch in range(start_epoch, epochs+1):
    epochs_count.append(epoch)

    print("* Training epoch {}:".format(epoch))
    epoch_time, epoch_loss, epoch_accuracy, epoch_AUC = train(model, train_loader, optimizer, criterion, epoch, max_gradient_norm)

    train_losses.append(epoch_loss)
    print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%, AUC: {:.4f}\n"
          .format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_AUC))

    print("* Validation for epoch {}:".format(epoch))
    epoch_time, epoch_loss, epoch_accuracy, epoch_AUC = validate(model,
                              dev_loader,
                              criterion)

    valid_losses.append(epoch_loss)
    print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, AUC: {:.4f}\n"
          .format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_AUC))

    scheduler.step(epoch_accuracy)
      
    if epoch_accuracy < best_score:
      patience_counter += 1
    else:
      best_score = epoch_accuracy
      patience_counter = 0

      torch.save({"epoch": epoch,
            "model": model.state_dict(),
            "best_score": best_score,
            "epochs_count": epochs_count,
            "train_losses": train_losses,
            "valid_losses": valid_losses},
          os.path.join(target_dir, "best.pth.tar"))

    torch.save({"epoch": epoch,
          "model": model.state_dict(),
          "best_score": best_score,
          "optimizer": optimizer.state_dict(),
          "epochs_count": epochs_count,
          "train_losses": train_losses,
          "valid_losses": valid_losses},
          os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

    if patience_counter >= patience:
      print("-> Early stopping: patience limit reached, stopping...")
      break

if __name__ == '__main__':
  print("start training")
  training(checkpoint)
      

