import pandas as pd
import os, re, pickle

from miditoolkit.midi import parser
from utils import tempo2sec
from remi_containers import BeatREMI
from containers import NoteMCSV, ChordMCSV
from chord_processor import ChordProcessor
from mcsv_to_midi import Song

class MidiDecoder(object):
  def __init__(self, event_csv, chord_processor, read_csv=True, events=None, max_duration=None, transfer_to_full_event=False, vocab=None):
    if read_csv:
      if not transfer_to_full_event:
        self.events = pd.read_csv(event_csv, encoding='utf-8')['EVENT'].tolist()
      else:
        assert vocab is not None
        ev_enc = pd.read_csv(event_csv, encoding='utf-8')['ENCODING'].tolist()
        self.events = [vocab.idx2event[ev] for ev in ev_enc]
    else:
      self.events = events

    self.chord_processor = chord_processor
    self.allowed_tempo_pos = [0, 16, 32, 48]
    self.max_duration = max_duration
    self.max_bars = 65536

  def get_beats(self):
    beat_n_tempos = [ev for ev in self.events if re.match('Bar|Position_*|Tempo-Class_*|Tempo_*', ev)]
    self.beat_dict, self.n_bars = {}, 0

    cur_barpos = [0, 0]
    cur_start_time = 0.2

    for i, ev in enumerate(beat_n_tempos):
      if ev == 'Bar':
        if self.max_duration and cur_start_time > self.max_duration:
          self.max_bars = self.n_bars
          break
        self.n_bars += 1
      elif 'Position' in ev:
        cur_barpos = [self.n_bars, int(ev.split('_')[-1].split('/')[0])]

      elif 'Tempo-Class' in ev:
        if cur_barpos[1] not in self.allowed_tempo_pos:
          continue
        assert 'Tempo_' in beat_n_tempos[i+1], 'illegal event following tempo class: {}'.format(beat_n_tempos[i+1])

        beat_duration = tempo2sec( float( beat_n_tempos[i+1].split('_')[-1] ) )
        self.beat_dict[ (cur_barpos[0], cur_barpos[1] // 16) ] = \
          BeatREMI(cur_barpos[1] == 0, cur_barpos[0], cur_barpos[1], cur_start_time, beat_duration)
        cur_start_time += beat_duration

    self.max_duration = cur_start_time

    return
  
  def get_chords(self):
    beat_n_chords = [ev for ev in self.events if re.match('Bar|Position_*|Chord-Tone_*|Chord-Type_*|Chord-Slash_*', ev)]
    beat_has_chord = set()
    self.chords = []

    cur_barbeat = [0, 0]
    for i, ev in enumerate(beat_n_chords):
      if ev == 'Bar':
        cur_barbeat[0] += 1
        if cur_barbeat[0] > self.max_bars:
          break
      elif 'Position' in ev:
        cur_barbeat[1] = int(ev.split('_')[-1].split('/')[0]) // 16
      elif 'Chord-Tone' in ev:
        if tuple(cur_barbeat) in beat_has_chord:
          continue
        assert 'Chord-Type' in beat_n_chords[i+1] and 'Chord-Slash' in beat_n_chords[i+2]

        if self.chords:
          self.chords[-1].duration_sec = self.beat_dict[ tuple(cur_barbeat) ].start_time - self.chords[-1].onset_sec

        ch_type = beat_n_chords[i+1].split('_')[-1] if beat_n_chords[i+1] != 'Chord-Type_' else ''
        tone, ch_type, slash = ev.split('_')[-1], ch_type, beat_n_chords[i+2].split('_')[-1]
        chord_notes, bass = self.chord_processor.compute_notes(None, require_parsing=False, tone=tone, ch_type=ch_type, slash=slash)

        self.chords.append(
          ChordMCSV(ch_type, bass, chord_notes, None, self.beat_dict[ tuple(cur_barbeat) ].start_time, None)
        )
        beat_has_chord.add( tuple(cur_barbeat) )

    if self.chords and self.max_duration:
      self.chords[-1].duration_sec = self.max_duration - self.chords[-1].onset_sec
    elif self.chords:
      self.chords[-1].duration_sec = self.beat_dict[ (self.n_bars - 1, 3) ].start_time + self.beat_dict[ (self.n_bars - 1, 3) ].duration - self.chords[-1].onset_sec

    return

  def get_notes(self):
    beat_n_notes = [ev for ev in self.events if re.match('Bar|Position_*|Note-Velocity_*|Note-On_*|Note-Duration_*', ev)]
    self.notes = []

    cur_barpos = [0, 0]
    for i, ev in enumerate(beat_n_notes):
      if ev == 'Bar':
        cur_barpos[0] += 1
        if cur_barpos[0] > self.max_bars:
          break
      elif 'Position' in ev:
        cur_barpos[1] = int(ev.split('_')[-1].split('/')[0])
      elif 'Note-Velocity' in ev:
        assert 'Note-On' in beat_n_notes[i+1] and 'Note-Duration' in beat_n_notes[i+2], 'illegal event following note vel: {} {}'.format(beat_n_notes[i+1], beat_n_notes[i+2])

        cur_barbeat = (cur_barpos[0], cur_barpos[1] // 16)
        note_velocity = int(ev.split('_')[-1]) * 4 + 3
        note_pitch = int(beat_n_notes[i+1].split('_')[-1])
        note_duration = int(beat_n_notes[i+2].split('_')[-1].split('/')[0])
        onset_sec, duration_sec = self._compute_note_onset_duration_sec(cur_barpos[1], note_duration, cur_barbeat)

        self.notes.append(
          NoteMCSV(note_pitch, note_velocity, onset_sec, duration_sec)
        )

    return

  def get_structure_info(self):
    beat_n_structs = [ev for ev in self.events if re.match('Bar|Position_.*|Part-.*|Rep-.*|Phrase|MLU-.*|Backref-.*', ev)]
    self.structs = []

    cur_barpos = [0, 0]
    for i, ev in enumerate(beat_n_structs):
      if ev == 'Bar':
        cur_barpos[0] += 1
        if cur_barpos[0] > self.max_bars:
          break
      elif 'Position' in ev:
        cur_barpos[1] = int(ev.split('_')[-1].split('/')[0])
      else:
        if cur_barpos[0] > 0:
          cur_barbeat = (cur_barpos[0], cur_barpos[1] // 16)
        else:
          cur_barbeat = (1, 0)
          
        onset_sec, _ = self._compute_note_onset_duration_sec(cur_barpos[1], 0., cur_barbeat)

        self.structs.append(
          {'EVENT': ev, 'TIMESTAMP': round(onset_sec, 3)}
        )

  def patch_chord_velocity(self, relative_vel=0.5):
    if not self.chords:
      print ('no chords to process, skipping ...')
    assert self.notes is not None

    cur_melody_idx = 0
    moving_vel = 40

    for i, ch in enumerate(self.chords):
      melody_vels = []
      while cur_melody_idx < len(self.notes) and self.notes[cur_melody_idx].onset_sec < ch.onset_sec + ch.duration_sec:
        melody_vels.append( self.notes[cur_melody_idx].velocity )
        cur_melody_idx += 1
      
      if not melody_vels:
        velocity = self.chords[i-1].velocity if i > 0 else 40
      else:
        velocity = int(relative_vel * sum(melody_vels) / len(melody_vels))

      moving_vel = int(0.5 * moving_vel + 0.5 * velocity)
      ch.velocity = moving_vel
      self.chords[i] = ch

    return

  def write_to_midi(self, output_file, chord_note_offset=0.05, tempo=120, chord_instrument=33):
    Song.write_to_midi(self, output_file, chord_note_offset=chord_note_offset, tempo=tempo, chord_instrument=chord_instrument)
    return
  
  def _compute_note_onset_duration_sec(self, note_pos, note_duration, ref_barbeat):
    onset_sec = \
      self.beat_dict[ ref_barbeat ].start_time + \
      (note_pos - ref_barbeat[1] * 16) / 16 * self.beat_dict[ ref_barbeat ].duration

    duration_sec = note_duration / 16 * self.beat_dict[ ref_barbeat ].duration
    
    return onset_sec, duration_sec

def convert_events_to_midi(event_source, output_midi, chord_processor, use_structure=False, output_struct_csv=None, transfer_to_full_event=False, vocab=None, max_duration=None):
  if isinstance(event_source, list):
    midi_dec = MidiDecoder(None, chord_processor, read_csv=False, events=event_source, max_duration=max_duration)
    print ('>> now converting events to MIDI:', output_midi, '...')
  else:
    midi_dec = midi_dec = MidiDecoder(event_source, chord_processor, transfer_to_full_event=transfer_to_full_event, vocab=vocab, max_duration=max_duration)

  midi_dec.get_beats()
  midi_dec.get_chords()
  midi_dec.get_notes()
  midi_dec.patch_chord_velocity()
  midi_dec.write_to_midi(output_midi)

  if use_structure:
    midi_dec.get_structure_info()
    df_structs = pd.DataFrame(midi_dec.structs, columns=['EVENT', 'TIMESTAMP'])
    df_structs.to_csv(output_struct_csv, encoding='utf-8', index=False)

  return

if __name__ == '__main__':
  ch_proc = pickle.load(open('./pickles/chord_processor.pkl', 'rb'))
  midi_dec = MidiDecoder('test_out.csv', ch_proc)
  midi_dec.get_beats()

  for k in sorted(midi_dec.beat_dict.keys(), key=lambda x: (x[0], x[1])):
    print (k, midi_dec.beat_dict[k].pos_remi)
    print (midi_dec.beat_dict[k].is_bar, round(midi_dec.beat_dict[k].start_time, 3), round(midi_dec.beat_dict[k].duration, 3))

  midi_dec.get_chords()
  for ch in midi_dec.chords:
    print (ch)

  midi_dec.get_notes()
  for note in midi_dec.notes:
    print (note)

  midi_dec.patch_chord_velocity()
  for ch in midi_dec.chords:
    print (ch)

  # midi_dec.write_to_midi('test_out.midi', chord_instrument=33)