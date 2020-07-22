# convert csv(original) to remi events (.csv format)
import pandas as pd
import numpy as np
import pickle, os
from glob import glob
from functools import cmp_to_key
import matplotlib.pyplot as plt

from chord_processor import ChordProcessor
from mlu_processor import MLUProcessor
from build_vocab import Vocab
from remi_containers import NoteREMI, ChordREMI, BeatREMI, sort_remi_events_cmp
from explore_mcsv import get_chords_info
from utils import clip_val, db2velocity
# [read]
def collect_beats(beat_df, vocab):
  beat_dict = {}
  for i, beat in beat_df.iterrows():
    # BeatREMI (is_bar, bar, position, start_time, duration):
    beat_dict[(int(beat['bar']), int(beat['beat']))] = \
      BeatREMI(beat['beat'] == 1, int(beat['bar']), (int(beat['beat']) - 1) * 16, beat['onset'], beat['duration'])
    
    beat_dict[(int(beat['bar']), int(beat['beat']))].get_tempo(vocab.tempo_cls_bounds, vocab.tempo_vals)

  return beat_dict

def collect_segments(beat_df, beat_dict):
  cur_seg = 'NA'

  for i, beat in beat_df.iterrows():
    if beat['form'] != cur_seg and i != len(beat_df) - 1:
      beat_key = (int(beat['bar']), int(beat['beat']))
      if cur_seg != 'NA':
        beat_dict[beat_key].patch_segment_tag(end_seg=cur_seg, start_seg=beat['form'])
      else:
        beat_dict[beat_key].patch_segment_tag(start_seg=beat['form'])
      cur_seg = beat['form']
    
    if i == len(beat_df) - 1:
      beat_key = (int(beat['bar']), int(beat['beat']))
      beat_dict[beat_key].patch_segment_tag(end_seg=cur_seg)

  return beat_dict        

# [read]
def collect_chords(beat_df, chord_processor):
  chords = get_chords_info(beat_df, repeat_long_chord=True, repeat_beats=4)
  chords_remi = []

  for ch in chords:
    if ch.id == 'NC':
      continue
    else:
      # ex: Bb6 -> tone = 'Bb' typ = '6' slash = ''
      tone, typ, slash = chord_processor._parse_chord_literal(ch.id)

    if not slash:
      # ex: Bb6 -> tone = 'Bb' typ = '6' slash = 'Bb'
      slash = tone

    # ch format = =(self, id, start_time, end_time, start_barbeat, end_barbeat)
    # start_barbeat[0] : bar
    # end_barbeat[1] : beat
    # 
    # ChordREMI def __init__(self, tone, typ, slash, bar, position)
    # 
    # (ch.start_barbeat[1] - 1) * 16 
    # 64 slots for 1 bar, 4 beats for 1 bar
    # 1 beat = 16 slots
    chords_remi.append(
      ChordREMI(tone, typ, slash, ch.start_barbeat[0], (ch.start_barbeat[1] - 1) * 16)
    )

  return chords_remi

# [read]
def collect_notes(melody_df, beat_dict, default_vel=20):
  notes = []
  n_short = 0

  for i, note in melody_df.iterrows():
    if note['duration'] < 1e-3:
      n_short += 1

    beat = beat_dict[(int(note['bar']), int(note['beat']))]
    beat_dur = beat.duration

    # quantize note_duration
    note_dur = round(note['duration'] / beat_dur * 16)
    note_dur = clip_val(note_dur, 1, 32)

    note_pos = round((note['onset'] - beat.start_time) / beat.duration * 16) + beat.pos_remi.position
    note_pos = clip_val(note_pos, 0, 63)

    if 'loud_median' in melody_df.columns and not np.isnan(note['loud_median']):
      note_vel = db2velocity(abs(note['loud_median'])) // 4
    else:
      note_vel = default_vel

    notes.append(
      NoteREMI(int(note['pitch']), note_vel, int(note['bar']), note_pos, note_dur)
    )

  return notes, n_short

def collect_mlus(melody_df, notes, mlu_processor):
  # fill up empty
  if 'phrase_id' in melody_df.columns:
    assert 'phrase_begin' in melody_df.columns
  if 'phrase_begin' not in melody_df.columns:
    melody_df['phrase_begin'] = 0
  if 'idea' not in melody_df.columns:
    melody_df['idea'] = 'NA'

  cur_idea = 'NA'
  mlu_pos = []
  for i, note in melody_df.iterrows():
    # for matching reson
    assert notes[i].pitch == note['pitch']

    if note['idea'] != cur_idea:
      cur_idea = note['idea']
      is_phrase = bool(note['phrase_begin'])
      has_void, rep_backref, rep_variation, typ, sub_typ = mlu_processor.parse_mlu_literal(note['idea'])

      if rep_backref:
        if len(mlu_pos) < rep_backref:
          rep_backref, rep_variation = '', ''
        else:
          notes[ mlu_pos[-rep_backref] ].mlu_tag.referred = True

      notes[i].patch_mlu_tag(is_phrase, has_void, rep_backref, rep_variation, typ, sub_typ)
      mlu_pos.append(i)

  return notes

def event_to_encodings(events, vocab, chord_processor, use_structure=False):
  rows = []
  # [question] appeared_pos use?
  appeared_pos = set()

  for i, ev in enumerate(events):
    if ev.ev_type == 'beat':
      if use_structure and ev.segment_tag is not None and i != len(events) - 1:
        if ev.segment_tag.part_end:
          rows.append( {'EVENT': 'Rep-End', 'VALUE': ev.segment_tag.rep_end, 'VALUE_IDX': ev.segment_tag.rep_end, 'ENCODING': vocab.event2idx['Rep-End_{}'.format(ev.segment_tag.rep_end)]} )
          rows.append( {'EVENT': 'Part-End', 'VALUE': ev.segment_tag.part_end, 'VALUE_IDX': ev.segment_tag.part_end, 'ENCODING': vocab.event2idx['Part-End_{}'.format(ev.segment_tag.part_end)]} )
        if ev.segment_tag.part_start:
          rows.append( {'EVENT': 'Part-Start', 'VALUE': ev.segment_tag.part_start, 'VALUE_IDX': ev.segment_tag.part_start, 'ENCODING': vocab.event2idx['Part-Start_{}'.format(ev.segment_tag.part_start)]} )
          rows.append( {'EVENT': 'Rep-Start', 'VALUE': ev.segment_tag.rep_start, 'VALUE_IDX': ev.segment_tag.rep_start, 'ENCODING': vocab.event2idx['Rep-Start_{}'.format(ev.segment_tag.rep_start)]} )

      if ev.is_bar:
        appeared_pos = set()
        rows.append( {'EVENT': 'Bar', 'VALUE': 0, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Bar']} )

      appeared_pos.add( ev.pos_remi.position )
      rows.append( {'EVENT': 'Position', 'VALUE': ev.pos_remi.position, 'VALUE_IDX': ev.pos_remi.position, 'ENCODING': vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)]})
      rows.append( {'EVENT': 'Tempo-Class', 'VALUE': ev.tempo_cls, 'VALUE_IDX': ev.tempo_cls, 'ENCODING': vocab.ev_type_base['Tempo-Class'] + ev.tempo_cls} )
      rows.append( {'EVENT': 'Tempo', 'VALUE': vocab.idx2event[ vocab.ev_type_base['Tempo'] + ev.tempo_bin ].split('_')[-1], 'VALUE_IDX': ev.tempo_bin, 'ENCODING': vocab.ev_type_base['Tempo'] + ev.tempo_bin} )

      if use_structure and ev.segment_tag is not None and i == len(events) - 1:
        rows.append( {'EVENT': 'Rep-End', 'VALUE': ev.segment_tag.rep_end, 'VALUE_IDX': ev.segment_tag.rep_end, 'ENCODING': vocab.event2idx['Rep-End_{}'.format(ev.segment_tag.rep_end)]} )
        rows.append( {'EVENT': 'Part-End', 'VALUE': ev.segment_tag.part_end, 'VALUE_IDX': ev.segment_tag.part_end, 'ENCODING': vocab.event2idx['Part-End_{}'.format(ev.segment_tag.part_end)]} )

    elif ev.ev_type == 'chord':
      if ev.pos_remi.position not in appeared_pos:
        appeared_pos.add( ev.pos_remi.position )
        rows.append( {'EVENT': 'Position', 'VALUE': ev.pos_remi.position, 'VALUE_IDX': ev.pos_remi.position, 'ENCODING': vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)]})
      
      tone_idx, slash_idx = chord_processor.key_map[ev.tone], chord_processor.key_map[ev.slash]
      rows.append( {'EVENT': 'Chord-Tone', 'VALUE': ev.tone, 'VALUE_IDX': tone_idx, 'ENCODING': vocab.event2idx['Chord-Tone_{}'.format(ev.tone)]})
      rows.append( {'EVENT': 'Chord-Type', 'VALUE': ev.typ, 'VALUE_IDX': 'None', 'ENCODING': vocab.event2idx['Chord-Type_{}'.format(ev.typ)]} )
      rows.append( {'EVENT': 'Chord-Slash', 'VALUE': ev.slash, 'VALUE_IDX': slash_idx, 'ENCODING': vocab.event2idx['Chord-Slash_{}'.format(ev.slash)]} )
      
    elif ev.ev_type == 'note':
      if ev.pos_remi.position not in appeared_pos:
        appeared_pos.add( ev.pos_remi.position )
        rows.append( {'EVENT': 'Position', 'VALUE': ev.pos_remi.position, 'VALUE_IDX': ev.pos_remi.position, 'ENCODING': vocab.event2idx['Position_{}/64'.format(ev.pos_remi.position)]})
      
      if use_structure and ev.mlu_tag is not None:
        if ev.mlu_tag.is_phrase:
          rows.append( {'EVENT': 'Phrase', 'VALUE': 0, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Phrase']} )
        
        rows.append( {'EVENT': 'MLU-Type', 'VALUE': ev.mlu_tag.typ, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['MLU-Type_{}'.format(ev.mlu_tag.typ)]} )
        if ev.mlu_tag.sub_typ:
          rows.append( {'EVENT': 'MLU-Subtype', 'VALUE': ev.mlu_tag.sub_typ, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['MLU-Subtype_{}-{}'.format(ev.mlu_tag.typ, ev.mlu_tag.sub_typ)]} )
        elif not ev.mlu_tag.sub_typ and 'MLU-Subtype_{}-general'.format(ev.mlu_tag.typ) in vocab.event2idx.keys():
          rows.append( {'EVENT': 'MLU-Subtype', 'VALUE': 'general', 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['MLU-Subtype_{}-general'.format(ev.mlu_tag.typ)]} )

        if ev.mlu_tag.rep_backref:
          rows.append( {'EVENT': 'Backref-Distance', 'VALUE': ev.mlu_tag.rep_backref, 'VALUE_IDX': int(ev.mlu_tag.rep_backref), 'ENCODING': vocab.event2idx['Backref-Distance_{}'.format(ev.mlu_tag.rep_backref)]} )
        if ev.mlu_tag.rep_variation:
          rows.append( {'EVENT': 'Backref-Variation', 'VALUE': ev.mlu_tag.rep_variation, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Backref-Variation_{}'.format(ev.mlu_tag.rep_variation)]} )
        
        if ev.mlu_tag.referred:
          rows.append( {'EVENT': 'Backref-Referred', 'VALUE': 0, 'VALUE_IDX': 0, 'ENCODING': vocab.event2idx['Backref-Referred']} )

      rows.append( {'EVENT': 'Note-Velocity', 'VALUE': ev.velocity, 'VALUE_IDX': ev.velocity, 'ENCODING': vocab.event2idx['Note-Velocity_{}'.format(ev.velocity)]})
      rows.append( {'EVENT': 'Note-On', 'VALUE': ev.pitch, 'VALUE_IDX': ev.pitch, 'ENCODING': vocab.event2idx['Note-On_{}'.format(ev.pitch)]} )
      rows.append( {'EVENT': 'Note-Duration', 'VALUE': ev.duration, 'VALUE_IDX': ev.duration, 'ENCODING': vocab.event2idx['Note-Duration_{}/64'.format(ev.duration)]} )

  df_enc = pd.DataFrame(rows, columns=['EVENT', 'VALUE', 'VALUE_IDX', 'ENCODING'])
  print (df_enc.info())

  return df_enc

def convert_piece(beat_csv, melody_csv, out_csv, vocab, chord_processor, accept_timesig=['\'4/4\''], use_structure=False, mlu_processor=None):
  beat_df = pd.read_csv(beat_csv, encoding='utf-8')
  melody_df = pd.read_csv(melody_csv, encoding='utf-8')
  
  # only accept 4/4
  if beat_df['signature'].unique().tolist() != accept_timesig:
    print ('[Info] Invalid time signature: {}, skipping ...'.format(beat_df['signature'].unique().tolist()))
    return 0

  beat_dict = collect_beats(beat_df, vocab)
  if use_structure and 'form' in beat_df.columns:
    beat_dict = collect_segments(beat_df, beat_dict)

  chords = collect_chords(beat_df, chord_processor)
  notes, n_short = collect_notes(melody_df, beat_dict)
  if use_structure:
    notes = collect_mlus(melody_df, notes, mlu_processor)

  beats = [beat_dict[k] for k in sorted(beat_dict.keys(), key=lambda x: (x[0], x[1]))]
  events = sorted(notes + chords + beats, key=cmp_to_key(sort_remi_events_cmp))

  df_enc = event_to_encodings(events, vocab, chord_processor, use_structure=use_structure)

  print ('>> writing to:', out_csv)
  df_enc.to_csv(out_csv, encoding='utf-8', index=False)

  return len(df_enc)

def convert_all_pieces(beat_csvs, melody_csvs, out_dir, vocab, chord_processor, use_structure=False, mlu_processor=None):
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  pieces_len = []
  idx = 1
  for bc, mc in zip(beat_csvs, melody_csvs):
    print ('file no:', idx)
    idx += 1
    print ('>> now processing:', mc)
    out_csv = os.path.join(out_dir, mc.replace('\\', '/').split('/')[-1].replace('.csv', '_remi.csv'))
    pieces_len.append( convert_piece(bc, mc, out_csv, vocab, chord_processor, use_structure=use_structure, mlu_processor=mlu_processor) )

  
  plt.hist(pieces_len, bins=20, range=(min(pieces_len), 6000), rwidth=0.6)
  plt.show()
  return

if __name__ == '__main__':
  beat_csvs = sorted( glob('../mcsv_beat/*.csv') )
  melody_csvs = sorted( glob('../mcsv_melody/*.csv') )
  vocab = pickle.load( open('../pickles/remi_wstruct_vocab.pkl', 'rb') )
  ch_proc = pickle.load( open('../pickles/chord_processor.pkl', 'rb') )
  mlu_proc =  pickle.load( open('../pickles/mlu_processor.pkl', 'rb') )

  # convert_piece(beat_csvs[317], melody_csvs[317], 'xxx', vocab, ch_proc, use_structure=True, mlu_processor=mlu_proc)

  convert_all_pieces(beat_csvs, melody_csvs, '../remi_encs_struct', vocab, ch_proc, use_structure=True, mlu_processor=mlu_proc)
  # for c in nd:
  #   print (c.pos_remi.bar, c.pos_remi.position, c.ev_type)
  #   if c.ev_type == 'note':
  #     print ('-- note:', c.pitch)