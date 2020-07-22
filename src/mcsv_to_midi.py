import random
import pickle, os, re
from glob import glob
import pandas as pd
import numpy as np

from miditoolkit.midi import parser
from miditoolkit.midi import TempoChange, Note, Instrument
from utils import sec2ticks, db2velocity
from explore_mcsv import get_chords_info
from chord_processor import ChordProcessor
from containers import ChordMCSV, NoteMCSV

class Song(object):
  def __init__(self, melody_csv, beat_csv, chord_processor):
    self.melody_df = pd.read_csv(melody_csv, encoding='utf-8')
    self.beat_df = pd.read_csv(beat_csv, encoding='utf-8')
    self.chord_processor = chord_processor

  def read_notes(self, default_vel=80):
    self.notes = []
    
    if 'loud_median' in self.melody_df.columns:
      for i, row in self.melody_df.iterrows():
        if row['duration'] < 5e-3:
          continue
        if np.isnan(row['loud_median']):
          note_vel = self.notes[-1].velocity if self.notes else default_vel
        else:
          note_vel = db2velocity(abs(row['loud_median']))

        self.notes.append(
          NoteMCSV(row['pitch'], note_vel, round(row['onset'], 3), round(row['duration'], 3))
        )
    else:
      for i, row in self.melody_df.iterrows():
        if row['duration'] < 5e-3:
          continue
        self.notes.append(
          NoteMCSV(row['pitch'], default_vel, round(row['onset'], 3), round(row['duration'], 3))
        )

    return

  def read_chords(self, relative_vel=0.6):
    assert self.notes is not None
    self.chords = []

    chords_seq = get_chords_info(self.beat_df)
    for c in chords_seq:
      print (' --', c)

    cur_melody_idx = 0
    for ch in chords_seq:
      chord_type = ch.id   
      if chord_type == 'NC':
        continue

      chord_notes, bass = self.chord_processor.compute_notes(ch.id)
      onset_sec, duration_sec = round(ch.start_time, 3), round(ch.end_time - ch.start_time, 3)

      melody_vels = []
      while cur_melody_idx < len(self.notes) and self.notes[cur_melody_idx].onset_sec < ch.end_time:
        melody_vels.append( self.notes[cur_melody_idx].velocity )
        cur_melody_idx += 1
      
      if not melody_vels:
        velocity = self.chords[-1].velocity if self.chords else 64
      else:
        velocity = int(relative_vel * sum(melody_vels) / len(melody_vels))

      self.chords.append( ChordMCSV(chord_type, bass, chord_notes, velocity, onset_sec, duration_sec) )

    return

  def write_to_midi(self, output_file, chord_note_offset=0.05, tempo=120, chord_instrument=33):
    midi_obj = parser.MidiFile()
    midi_obj.tempo_changes = [TempoChange(tempo, 0)]
    midi_obj.instruments = [
      Instrument(0, name='melody'), Instrument(chord_instrument, name='chords')
    ]

    for note in self.notes:
      # print ('processing note:', note)
      midi_obj.instruments[0].notes.append(
        Note(note.velocity, note.pitch, sec2ticks(note.onset_sec), sec2ticks(note.onset_sec + note.duration_sec))
      )

    for chord in self.chords:
      # print ('processing chord:', chord)
      if chord_note_offset * (len(chord.chord_notes)-1) > 0.5 * chord.duration_sec:
        offset = 0.5 * chord.duration_sec / (len(chord.chord_notes) - 1)
      else:
        offset = chord_note_offset

      midi_obj.instruments[1].notes.append(
        Note(chord.velocity, chord.bass, sec2ticks(chord.onset_sec), sec2ticks(chord.onset_sec + chord.duration_sec))
      )

      for i, n in enumerate(chord.chord_notes):
        note_onset_sec = chord.onset_sec + offset * i
        midi_obj.instruments[1].notes.append(
          Note(chord.velocity, n, sec2ticks(note_onset_sec), sec2ticks(chord.onset_sec + chord.duration_sec))
        )

    midi_obj.dump(output_file)
    return

def convert_all_mcsvs(chord_processor, melody_mcsv_dir, beat_mcsv_dir, output_dir):
  beat_csvs = sorted( glob( os.path.join(beat_mcsv_dir, '*.csv') ) )
  melody_csvs = sorted( glob( os.path.join(melody_mcsv_dir, '*.csv') ) )

  for bc, mc in zip(beat_csvs, melody_csvs):
    print ('>> converting: {} ...'.format(mc))
    assert bc.replace('\\', '/').split('/')[-1].split('_FINAL')[0] == mc.replace('\\', '/').split('/')[-1].split('_FINAL')[0]

    song = Song(mc, bc, chord_processor)
    song.read_notes()
    song.read_chords()

    output_file = os.path.join(output_dir, mc.replace('\\', '/').split('/')[-1].replace('.csv', '.midi'))
    print ('>> writing to: {} ...'.format(output_file))
    song.write_to_midi(output_file)

  return
    
if __name__ == '__main__':
  # beat_csvs = sorted( glob('mcsv_beat/*.csv') )
  # melody_csvs = sorted( glob('mcsv_melody/*.csv') )
  ch_proc = pickle.load(open('./pickles/chord_processor.pkl', 'rb'))
  convert_all_mcsvs(ch_proc, './mcsv_melody', './mcsv_beat', './midi_with_chords')

  # idx = random.choice([i for i in range(len(beat_csvs))])
  # print (melody_csvs[idx])
  # song = Song(melody_csvs[idx], beat_csvs[idx], ch_proc)
  # song.read_notes()
  # song.read_chords()

  # for n in song.notes:
  #   print (n)
  # for c in song.chords:
  #   print (c)

  # output_file = os.path.join('./midi_with_chords', melody_csvs[idx].split('\\')[-1].replace('.csv', '.midi'))
  # song.write_to_midi(output_file)