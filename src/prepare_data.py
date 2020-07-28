import sys, pickle
import numpy as np
import pandas as pd
from glob import glob

import random



def extract_seq_from_csv(enc_csv):
  return pd.read_csv(enc_csv, encoding='utf-8')['ENCODING'].tolist()


def make_training_data(enc_csv_paths):
  training_data = []
  piece_lens = []

  for csv_f in enc_csv_paths:
    print ('current_file > {}'.format(csv_f))
    words = extract_seq_from_csv(csv_f)
    training_data.append(words)
    

  return training_data



enc_csv_files = sorted( glob('../remi_encs_struct/*.csv') )
my_training_data = make_training_data(enc_csv_files)
print("Done loading csv")


print("Splitting validation set")
cutpoint = int(len(my_training_data)*0.95)
my_train = my_training_data[:cutpoint]
my_val = my_training_data[cutpoint:]
print("Training set count:{}".format(len(my_train)))
print("Validation set count:{}".format(len(my_val)))
# save training data to pickle file
train_pkl_filename = '../data/training_seqs_struct_new_final.pkl'
val_pkl_filename = '../data/val_seqs_struct_new_final.pkl'
pickle.dump(my_train, open(train_pkl_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(my_val, open(val_pkl_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

print("train set : {} val set : {}".format(train_pkl_filename,val_pkl_filename))