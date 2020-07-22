import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
from collections import Counter

from chord_processor import ChordProcessor
from utils import sec2tempo, chord_type, sort_seg_chord_cmp, sort_phrase_idea_cmp
from containers import Segment, StructEvent
from functools import cmp_to_key

import re

seg_counter, chord_counter = Counter(), Counter()
def update_cnt(base_cnt, new_cnt):
  base_cnt += new_cnt

def get_tempo_info(beat_df):
  mean, std = beat_df['duration'].mean(), beat_df['duration'].std()
  return sec2tempo(mean), sec2tempo( mean - 2*std ) - sec2tempo( mean + 2*std )

def get_note_duration_distr(beat_df, melody_df):
  beatlen_per_bar = beat_df.groupby(['bar'], as_index=False)['duration'].mean()
  bar_beatlen_dict = dict()

  for i, row in beatlen_per_bar.iterrows():
    bar_beatlen_dict[ int(row['bar']) ] = row['duration']

  log_durations = []
  for i, note in melody_df.iterrows():
    log_durations.append(
      np.log2( note['duration'] / bar_beatlen_dict[ note['bar'] ] )
    )

  return log_durations

def get_segment_info(beat_df):
  cur_seg = 'NA'
  segs = []

  for i, beat in beat_df.iterrows():
    if beat['form'] != cur_seg:
      if segs:
        segs[-1].end_time, segs[-1].end_bar = round(beat['onset'], 2), beat['bar']

      segs.append( Segment(beat['form'], round(beat['onset'], 2), None, beat['bar'], None ) )
      cur_seg = beat['form']

  segs[-1].end_time, segs[-1].end_bar = round(beat_df.loc[beat_df.index[-1], 'onset'], 2), beat_df.loc[beat_df.index[-1], 'bar']

  return segs

def get_chords_info(beat_df, repeat_long_chord=True, repeat_beats=8):
  cur_chord = 'NA'
  chords = []
  last_i = 0

  for i, beat in beat_df.iterrows():
    if beat['chord'] != cur_chord:
      if chords:
        # end_time = next entry's onset =  it's onset+duration 
        chords[-1].end_time, chords[-1].end_barbeat = round(beat['onset'], 3), (beat_df.loc[i-1, 'bar'], beat_df.loc[i-1, 'beat'])

      # StructEvent (id, start_time, end_time, start_barbeat, end_barbeat)
      chords.append( StructEvent(beat['chord'], round(beat['onset'], 3), None, (int(beat['bar']), int(beat['beat'])), None ) )
      cur_chord = beat['chord']
      last_i = i
    # add repeat_long_chord constraint for training purpose
    if repeat_long_chord:
      if i >= last_i + repeat_beats and beat['beat'] == 1:
        chords[-1].end_time, chords[-1].end_barbeat = round(beat['onset'], 3), (beat_df.loc[i-1, 'bar'], beat_df.loc[i-1, 'beat'])
        chords.append( StructEvent(beat['chord'], round(beat['onset'], 3), None, (int(beat['bar']), int(beat['beat'])), None ) )
        last_i = i     

  chords[-1].end_time, chords[-1].end_barbeat = round(beat_df.loc[beat_df.index[-1], 'onset'] + beat_df.loc[beat_df.index[-1], 'duration'] , 3), (beat_df.loc[beat_df.index[-1], 'bar'], beat_df.loc[beat_df.index[-1], 'beat'])

  return chords

def get_struct_event_info(melody_df, ev_type):
  cur_ev = 'NA'
  evs = []
  prefix = ev_type[:2].upper() + '-'

  for i, beat in melody_df.iterrows():
    if str(beat[ ev_type ]) != cur_ev:
      if evs:
        evs[-1].end_time, evs[-1].end_barbeat = round(beat['onset'], 3), (melody_df.loc[i-1, 'bar'], melody_df.loc[i-1, 'beat'])

      evs.append( StructEvent(prefix + str(beat[ ev_type ]), round(beat['onset'], 3), None, (beat['bar'], beat['beat']), None ) )
      cur_ev = str(beat[ ev_type ])

  evs[-1].end_time, evs[-1].end_barbeat = round(melody_df.loc[melody_df.index[-1], 'onset'] + melody_df.loc[melody_df.index[-1], 'duration'] , 3), (melody_df.loc[melody_df.index[-1], 'bar'], melody_df.loc[melody_df.index[-1], 'beat'])

  return evs


def get_unique_chords(beat_df):
  return beat_df['chord'].unique().tolist()

def get_unique_segments(beat_df):
  if 'form' not in beat_df.columns:
    return []
  
  return beat_df['form'].unique().tolist()

def get_unique_mlus(melody_df, remove_glue=True, remove_specifier=True):
  if 'idea' not in melody_df.columns:
    return []

  mlus = melody_df['idea'].unique().tolist()
  if remove_glue:
    mlus = [m.replace('~', '') for m in mlus]
  if remove_specifier:
    mlus = [re.sub(':.+', '', m) for m in mlus]
  
  mlus = [m.replace('*', '') for m in mlus]
  return mlus

def extract_beattrack_info(beat_csv):
  print ( 'Now processing:', beat_csv.replace('_FINAL_beattrack.csv', '') )
  print ('=====================================================')
  df = pd.read_csv(beat_csv, encoding='utf-8')

  print ('-- n_beats: \t{}'.format(len(df)))
  print ('-- signature: \t{}'.format( df.loc[0, 'signature']) )
  tempo, tempo_dev = get_tempo_info( df )
  print ('-- tempo: \t{:.2f} (+/- {:.2f}) bpm'.format(tempo, tempo_dev))

  segs = get_segment_info(df)
  print ('-- segments:')
  for seg in segs:
    print ('\t{}'.format(seg))

  chords = get_unique_chords(df)
  print ('-- chords:')
  print ('\t{}'.format(chords))
  print ('=====================================================')

  segs_unique = [ s.id.replace('\'', '') for s in segs ]
  chords_unique = [ chord_type(c) for c in chords ]

  chcnt, sgcnt = Counter(chords_unique), Counter(segs_unique)
  global seg_counter
  global chord_counter
  update_cnt(seg_counter, sgcnt)
  update_cnt(chord_counter, chcnt)

  chords = get_chords_info(df)

  return segs_unique, chords

def extract_note_info(melody_csv):
  print ('---------- melody info ----------')
  df = pd.read_csv(melody_csv, encoding='utf-8')
  print ('-- n_notes: \t{}'.format( len(df )))

  if 'loud_median' in df.columns:
    note_strengths = df['loud_median'].apply(lambda x: abs(x))
    print ('-- min_db: \t{:.2f}'.format(note_strengths.min()))
    print ('-- max_db: \t{:.2f}'.format(note_strengths.max()))
    print ('-- overall_strength: \t{:.2f} (+/- {:.2f}) db'.format(note_strengths.mean(), 2 * note_strengths.std()))

  return

def extract_structural_info(beat_csv, melody_csv):
  print ('>> now processing: {}'.format(melody_csv))
  print ('')
  df_beat = pd.read_csv(beat_csv, encoding='utf-8')
  df_melody = pd.read_csv(melody_csv, encoding='utf-8')

  segs = get_segment_info(df_beat)
  chords = get_chords_info(df_beat, repeat_long_chord=False)

  segs_chords = sorted(segs + chords, key=cmp_to_key(sort_seg_chord_cmp))
  print ('****** ------------ structure & chords ------------ ******')
  for x in segs_chords:
    if isinstance(x, Segment):
      print ('==========================================================')
      print ('  ** Segment {}: {} ~ {} sec / bar {} ~ {} **'.format(x.id, x.start_time, x.end_time, x.start_bar, x.end_bar))
      print ('==========================================================')
    else:
      print ('  ----> {}: {} ~ {} sec / (bar, beat) {} ~ {}'.format(x.id, x.start_time, x.end_time, x.start_barbeat, x.end_barbeat))

  print ('')
  phrases = get_struct_event_info(df_melody, 'phrase_id')
  mlus = get_struct_event_info(df_melody, 'idea')
  phrases_mlus = sorted(phrases + mlus, key=cmp_to_key(sort_phrase_idea_cmp))
  print ('****** -------------- phrases & MLUs -------------- ******')
  for x in phrases_mlus:
    if 'PH' in x.id:
      print ('')
      print ('  ** [[Phrase]] {}: {} ~ {} sec / (bar, beat) {} ~ {} **'.format(x.id.split('-')[-1], x.start_time, x.end_time, x.start_barbeat, x.end_barbeat))
    else:
      print ('    --> {}: {} ~ {} sec / (bar, beat) {} ~ {}'.format(x.id.split('-')[-1], x.start_time, x.end_time, x.start_barbeat, x.end_barbeat))

  return

if __name__ == '__main__':
  beat_csvs = sorted( glob('mcsv_beat/*.csv') )
  melody_csvs = sorted( glob('mcsv_melody/*.csv') )
  # print (len(beat_csvs))
  # beat_csvs = random.sample(beat_csvs, 20)

  # idx = random.choice([x for x in range(len(beat_csvs))])
  # extract_structural_info(beat_csvs[idx], melody_csvs[idx])

  ''' for getting unique mlus '''
  # mlu_counter = Counter()
  # for i, mc in enumerate(melody_csvs):
  #   print (i)
  #   new_cnt = Counter( get_unique_mlus(pd.read_csv(mc, encoding='utf-8')) )
  #   mlu_counter += new_cnt

  # for x in sorted(mlu_counter.items(), key=lambda x: x[1], reverse=True):
  #   print ('{} \t cnt: {}'.format(x[0], x[1]))
  # print (len(mlu_counter))
  
  ''' for note length check '''
  # notelens = []
  # for bc, mc in zip(beat_csvs, melody_csvs):
  #   print (bc, mc)
  #   notelens.extend(get_note_duration_distr(pd.read_csv(bc, encoding='utf-8'), pd.read_csv(mc, encoding='utf-8')))

  # plt.hist(notelens, range=(-5, 2), bins=20, rwidth=0.6)
  # plt.show()
  
  ''' for examining unique chords & segments '''
  # ch_proc = ChordProcessor('./pickles/chord_profile.pkl', './pickles/key_map.pkl')
  # for ch in ch_set:
  #   if ch == 'NC':
  #     continue
  #   ch_proc.compute_notes(ch)

  # print ('SEGMENT IDS:')
  # for s in sorted(segs_set):
  #   print(' ', s)
  
  # print ('CHORD TYPES:')
  # cnt = 1
  # for c in sorted(chords_set):
  #   print(cnt, ':', c)
  #   cnt += 1

  # print ('Segments:')
  # print ('-----------------------------------------------------')
  # for s, cnt in sorted(seg_counter.items(), key=lambda x: x[1], reverse=True):
  #   print (s, ':', cnt)


  # print ('-----------------------------------------------------')
  # print ('Chords:')
  # print ('-----------------------------------------------------')
  # for c, cnt in sorted(chord_counter.items(), key=lambda x: x[1], reverse=True):
  #   print (c, ':', cnt)

  segs_counter = Counter()
  for i, bc in enumerate(beat_csvs):
    print (i)
    new_cnt = Counter( get_unique_segments(pd.read_csv(bc, encoding='utf-8')) )
    segs_counter += new_cnt

  for x in sorted(segs_counter.items(), key=lambda x: x[1], reverse=True):
    print ('{} \t cnt: {}'.format(x[0], x[1]))
  print (len(segs_counter))

  