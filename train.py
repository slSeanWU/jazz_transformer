import argparse
import sys, pickle , os

parser = argparse.ArgumentParser()

parser.add_argument('ckpt_dir' , help="the folder to save checkpoints")
parser.add_argument('log_file' , help="the file path to save log file")

args = parser.parse_args()

sys.path.append('./transformer_xl/')
sys.path.append('./src/')



import numpy as np
import pandas as pd
from glob import glob
from build_vocab import Vocab

# which gpu to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model_aug import TransformerXL



if __name__ == '__main__':

  # load dictionary
  # generated from build_vocab.py
  vocab = pickle.load(open('pickles/remi_wstruct_vocab.pkl', 'rb'))
  event2word, word2event = vocab.event2idx, vocab.idx2event
  
  
  # load train data
  # training_seqs_final.pkl : all songs' remi format
  training_data_file = "data/training_seqs_struct_new_final.pkl"
  print("loading training data from {}".format(training_data_file))
  training_seqs = pickle.load( open(training_data_file, 'rb') )

  # show size of trqaining data
  print("Training data count: {}".format(len(training_seqs)))


  # declare model
  model = TransformerXL(
      event2word=event2word, 
      word2event=word2event,
      checkpoint=None,
      is_training=True,
      training_seqs=training_seqs)
  
  # train
  model.train_augment(output_checkpoint_folder=args.ckpt_dir, logfile=args.log_file)
  
  # close
  model.close()
