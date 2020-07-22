from functools import total_ordering
import numpy as np

from utils import clip_val

@total_ordering
class PositionREMI(object):
  def __init__(self, bar, position):
    self.bar = bar
    self.position = position

  def __eq__(self, other):
    return self.bar == other.bar and self.position == other.position

  def __lt__(self, other):
    # use for compare before and after 
    if self.bar != other.bar:
      return self.bar < other.bar
    
    return self.position < other.position

  def __repr__(self):
    return '[REMI pos] bar: {} | pos: {}'.format(self.bar, self.position)

class BaseEventREMI(object):
  def __init__(self, ev_type, bar, position):
    self.ev_type = ev_type
    self.pos_remi = PositionREMI(bar, position)

class ChordREMI(BaseEventREMI):
  def __init__(self, tone, typ, slash, bar, position):
    super().__init__('chord', bar, position)
    self.tone = tone
    self.typ = typ
    self.slash = slash

class NoteREMI(BaseEventREMI):
  def __init__(self, pitch, velocity, bar, position, duration):
    super().__init__('note', bar, position)
    self.pitch = pitch
    self.velocity = velocity
    self.duration = duration
    self.mlu_tag = None

  def __repr__(self):
    return '[Note]\n  -- {}\n  -- pitch: {}, velocity: {}, duration: {}\n  -- {}\n===================='.format(
      self.pos_remi, self.pitch, self.velocity, self.duration, self.mlu_tag
    )

  def patch_mlu_tag(self, is_phrase, has_void, rep_backref, rep_variation, typ, sub_typ):
    self.mlu_tag = MLUTag(is_phrase, has_void, rep_backref, rep_variation, typ, sub_typ)

class BeatREMI(BaseEventREMI):
  def __init__(self, is_bar, bar, position, start_time, duration):
    super().__init__('beat', bar, position)
    self.is_bar = is_bar
    self.start_time = start_time
    self.duration = duration
    self.segment_tag = None

  def __repr__(self):
    return '[Beat]\n  -- is_bar: {}\n  -- {}\n  -- {}\n===================='.format(self.is_bar, self.pos_remi, self.segment_tag)

  def get_tempo(self, tempo_cls_bound, tempo_bins):
    # [question]
    self.tempo_cls = -1
    tempo = 1. / self.duration * 60 # beats per minutes

    for i, cl in enumerate(tempo_cls_bound):
      if tempo >= cl:
        self.tempo_cls = i

    self.tempo_cls = clip_val(self.tempo_cls, 0, len(tempo_cls_bound) - 2)
    self.tempo_bin = np.abs(tempo_bins - tempo).argmin()
  
  def patch_segment_tag(self, end_seg='', start_seg=''):
    part_start, part_end, rep_start, rep_end = '', '', '', ''

    end_seg, start_seg = end_seg.replace('\'', ''), start_seg.replace('\'', '')
    if end_seg:
      rep_end = end_seg[1] if len(end_seg) > 1 else '1'
      part_end = end_seg[0]
    if start_seg:
      rep_start = start_seg[1] if len(start_seg) > 1 else '1'
      part_start = start_seg[0]

    self.segment_tag = SegmentTag(part_end, rep_end, part_start, rep_start)
    

class SegmentTag(object):
  def __init__(self, part_end='', rep_end='', part_start='', rep_start=''):
    self.part_end = part_end
    self.rep_end = rep_end
    self.part_start = part_start
    self.rep_start = rep_start

  def __repr__(self):
    return '[segment] end: {:1}/{:1} | start: {:1}/{:1}'.format(self.part_end, self.rep_end, self.part_start, self.rep_start)

class MLUTag(object):
  def __init__(self, is_phrase, has_void, rep_backref, rep_variation, typ, sub_typ):
    self.is_phrase = is_phrase
    self.has_void = has_void
    self.rep_backref = rep_backref
    self.rep_variation = rep_variation
    self.typ = typ
    self.sub_typ = sub_typ
    self.referred = False

  def __repr__(self):
    return '[MLU] is_phrase: {:1} | has_void: {:1} | referred: {:1} | rep: {:1}/{:1} | type: {:10}/{:10}'.format(self.is_phrase, self.has_void, self.referred, self.rep_backref, self.rep_variation, self.typ, self.sub_typ)

def sort_remi_events_cmp(x, y):
  if x.pos_remi != y.pos_remi:
    if x.pos_remi < y.pos_remi:
      return -1
    else:
      return 1

  if isinstance(x, BeatREMI):
    return -1
  elif isinstance(x, ChordREMI):
    return -1
  else:
    if x.pitch < y.pitch:
      return -1
    else:
      return 1

if __name__ == '__main__':
  # test position comparison
  test_pos = [PositionREMI(3, 20), PositionREMI(3, 14), PositionREMI(1, 38)]
  test_pos = sorted(test_pos)
  for p in test_pos:
    print (p.bar, p.position)