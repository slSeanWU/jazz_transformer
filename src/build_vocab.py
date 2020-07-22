# assign ids to events
# output remi_wstruct_vocab.pkl
import pickle
import re
import numpy as np
from chord_processor import ChordProcessor
from mlu_processor import MLUProcessor

class Vocab(object):
  def __init__(self, tempo_cls_bounds, n_notevel_bins=32, n_notedur_bins=32, notedur_quantum=6, n_pos_bins=64, n_tempo_bins_per_cls=12, use_structure=False):
    self.event2idx = {}
    self.idx2event = {}
    self.idx_counter = 0 # reset counter

    self.tempo_cls_bounds = tempo_cls_bounds # the boundaries for tempo class
    self.n_notevel_bins = n_notevel_bins 
    self.n_notedur_bins = n_notedur_bins # the upper limit of the multuples of steps
    self.notedur_quantum = notedur_quantum # split 1 bar into 2^n steps
    self.n_pos_bins = n_pos_bins # split 1 bar into n_pos_bins
    self.n_tempo_bins_per_cls = n_tempo_bins_per_cls

    self.n_tempo_cls = len(tempo_cls_bounds) - 1
    self.tempo_vals = self._compute_tempo_vals() # all the quantized step for representing tempo
    self.notedurs = self._compute_notedurs()

    self.use_structure = use_structure

  # reuturn list of tempo steps
  # ex: [50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0, 82.5, 85.0, 87.5, 90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0, 112.5, 115.0, 117.5, 120.0, 122.5, 125.0, 127.5, 130.0, 132.5, 135.0, 137.5, 140.0, 143.33333333333334, 146.66666666666669, 150.00000000000003, 153.33333333333337, 156.6666666666667, 160.00000000000006, 163.3333333333334, 166.66666666666674, 170.00000000000009, 173.33333333333343, 176.66666666666677, 180.0, 191.66666666666666, 203.33333333333331, 214.99999999999997, 226.66666666666663, 238.3333333333333, 249.99999999999994, 261.66666666666663, 273.33333333333326, 284.9999999999999, 296.6666666666666, 308.33333333333326]
  def _compute_tempo_vals(self):

    tempo_vals = np.empty( (self.n_tempo_bins_per_cls * self.n_tempo_cls,) )

    # divide each class into sub steps
    for i in range(len(self.tempo_cls_bounds) - 1):
      low, high = self.tempo_cls_bounds[i], self.tempo_cls_bounds[i+1]
      tempo_vals[ i * self.n_tempo_bins_per_cls : (i+1) * self.n_tempo_bins_per_cls ] = \
        np.arange(low, high, (high - low) / self.n_tempo_bins_per_cls)

    return tempo_vals

  # return quantize steps of note duration
  # ex: ['1/64', '2/64', '3/64', '4/64', '5/64', '6/64', '7/64', '8/64', '9/64', '10/64', '11/64', '12/64', '13/64', '14/64', '15/64', '16/64', '17/64', '18/64', '19/64', '20/64', '21/64', '22/64', '23/64', '24/64', '25/64', '26/64', '27/64', '28/64', '29/64', '30/64', '31/64', '32/64']
  def _compute_notedurs(self):
    denom = str(2 ** self.notedur_quantum)
    notedurs = []

    # all steps (multiples of basic unit) (total n_notedur_bins ) (according to the notes duration distribution in the dataset)
    for i in range(1, self.n_notedur_bins + 1):
      notedurs.append( '{}/{}'.format(i, denom) )

    return notedurs

  # assign chord_processor and add chords in chord_profile to self.chord_types
  def add_chords(self, chord_processor):
    self.chord_processor = chord_processor
    self.chord_types = []

    for ch in self.chord_processor.chord_profile.keys():
      self.chord_types.append(ch)

    return

  # assign self.mlu_processor
  def add_mlus(self, mlu_processor):
    self.mlu_processor = mlu_processor
    return

  def build(self):
    # self.ev_type_base : the first event id for specific type of events

    # total 128 notes in midi
    
    # Note-On
    self.ev_type_base = {'Note-On': 0}
    for i in range(128):
      self.event2idx[ 'Note-On_{}'.format(i) ] = self.idx_counter
      self.idx_counter += 1

    # Note-Velocity
    self.ev_type_base['Note-Velocity'] = self.idx_counter
    for i in range(self.n_notevel_bins):
      self.event2idx[ 'Note-Velocity_{}'.format(i) ] = self.idx_counter
      self.idx_counter += 1

    # Note-Duration
    self.ev_type_base['Note-Duration'] = self.idx_counter
    for note_dur in self.notedurs:
      self.event2idx[ 'Note-Duration_{}'.format(note_dur) ] = self.idx_counter
      self.idx_counter += 1

    # Bar
    self.event2idx['Bar'] = self.idx_counter
    self.idx_counter += 1

    # Position
    self.ev_type_base['Position'] = self.idx_counter
    for i in range(self.n_pos_bins):
      self.event2idx[ 'Position_{}/{}'.format(i, self.n_pos_bins) ] = self.idx_counter
      self.idx_counter += 1

    # Tempo-Class
    self.ev_type_base['Tempo-Class'] = self.idx_counter
    for i in range(self.n_tempo_cls):
      self.event2idx[ 'Tempo-Class_{}'.format(i) ] = self.idx_counter
      self.idx_counter += 1

    # Tempo
    self.ev_type_base['Tempo'] = self.idx_counter
    for tempo in self.tempo_vals:
      self.event2idx[ 'Tempo_{:.2f}'.format(tempo) ] = self.idx_counter
      self.idx_counter += 1

    # chords
    if self.chord_types is not None:
      self.ev_type_base['Chord-Tone'] = self.idx_counter
      self.ev_type_base['Chord-Slash'] = self.idx_counter + 12
      for key, key_val in self.chord_processor.key_map.items():
        self.event2idx[ 'Chord-Tone_{}'.format(key) ] = self.idx_counter + key_val
        self.event2idx[ 'Chord-Slash_{}'.format(key) ] = self.idx_counter + key_val + 12

      self.idx_counter += 24

      self.ev_type_base['Chord-Type'] = self.idx_counter
      for ch in self.chord_types:
        self.event2idx[ 'Chord-Type_{}'.format(ch) ] = self.idx_counter
        self.idx_counter += 1

    # MLU
    if self.use_structure:
      self.event2idx['Phrase'] = self.idx_counter
      self.idx_counter += 1

      self.event2idx['MLU-Has-Void'] = self.idx_counter
      self.idx_counter += 1

      self.ev_type_base['MLU-Type'] = self.idx_counter
      for mlu_type in sorted(self.mlu_processor.mlu_types):
        self.event2idx[ 'MLU-Type_{}'.format(mlu_type) ] = self.idx_counter
        self.idx_counter += 1

      self.ev_type_base['MLU-Subtype'] = self.idx_counter
      for mlu_type in sorted(self.mlu_processor.mlu_subtypes.keys()):
        if len(self.mlu_processor.mlu_subtypes[ mlu_type ]) > 1:
          for st in sorted(self.mlu_processor.mlu_subtypes[ mlu_type ]):
            if not st:
              self.event2idx ['MLU-Subtype_{}-general'.format(mlu_type)] = self.idx_counter
            else:
              self.event2idx ['MLU-Subtype_{}-{}'.format(mlu_type, st)] = self.idx_counter
            self.idx_counter += 1
      
      self.event2idx['Backref-Referred'] = self.idx_counter
      self.idx_counter += 1

      self.ev_type_base['Backref-Distance'] = self.idx_counter
      for i in range(1, self.mlu_processor.max_back_ref + 1):
        self.event2idx['Backref-Distance_{}'.format(i)] = self.idx_counter
        self.idx_counter += 1
      
      self.ev_type_base['Backref-Variation'] = self.idx_counter
      for var in self.mlu_processor.rep_vars:
        if var:
          self.event2idx['Backref-Variation_{}'.format(var)] = self.idx_counter
          self.idx_counter += 1

      self.ev_type_base['Part-Start'] = self.idx_counter
      for part in ['A', 'B', 'C', 'D', 'I']:
        self.event2idx[ 'Part-Start_{}'.format(part) ] = self.idx_counter
        self.idx_counter += 1

      self.ev_type_base['Part-End'] = self.idx_counter
      for part in ['A', 'B', 'C', 'D', 'I']:
        self.event2idx[ 'Part-End_{}'.format(part) ] = self.idx_counter
        self.idx_counter += 1

      self.ev_type_base['Rep-Start'] = self.idx_counter
      for rep in [x for x in range(1, 7)]:
        self.event2idx[ 'Rep-Start_{}'.format(rep) ] = self.idx_counter
        self.idx_counter += 1

      self.ev_type_base['Rep-End'] = self.idx_counter
      for rep in [x for x in range(1, 7)]:
        self.event2idx[ 'Rep-End_{}'.format(rep) ] = self.idx_counter
        self.idx_counter += 1

    for k, v in self.event2idx.items():
      # exclude same events
      if not re.search('[A-G]b|E#|B#', k):
        self.idx2event[v] = k

    return

if __name__ == '__main__':
  vocab = Vocab([50, 80, 110, 140, 180, 320], use_structure=True)
  print (vocab.tempo_vals)
  print (vocab.notedurs)

  ch_proc = pickle.load( open('../pickles/chord_processor.pkl', 'rb') )
  mlu_proc = pickle.load( open('../pickles/mlu_processor.pkl', 'rb') )
  vocab.add_chords( ch_proc )
  vocab.add_mlus( mlu_proc )
  print (vocab.chord_types)

  vocab.build()
  for k, v in vocab.event2idx.items():
    print('{:03d}: {}'.format(v, k))
  for k, v in vocab.idx2event.items():
    print('{}: {:03d}'.format(v, k))

  pickle.dump(vocab, open('../pickles/remi_wstruct_vocab.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)