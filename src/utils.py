from containers import StructEvent, Segment

def chord_type(chord):
  chord = chord.split('/')[0]

  if chord == 'NC' or len(chord) < 2:
    return ''
  elif chord[1] in ['#', 'b']:
    return chord[2:]
  else:
    return chord[1:]

def sec2tempo(sec):
  return 60. / sec

def tempo2sec(bpm):
  return 60. / bpm

def sec2ticks(sec, tempo=120, ticks_per_beat=480):
  return int(sec * (tempo / 60.0) * ticks_per_beat)

def db2velocity(db, db_median=65, vel_10db_low=20, vel_10db_high=30):
  if db <= db_median:
    vel = int( 80 - vel_10db_low * (db_median - db) / 10 )
  else:
    vel = int( 80 + vel_10db_high * (db - db_median) / 10 )

  vel = min(120, max(10, vel))
  return vel

def sort_seg_chord_cmp(x, y):
  if x.start_time < y.start_time - 1e-2:
    return -1
  elif x.start_time > y.start_time + 1e-2:
    return 1
  elif isinstance(x, Segment):
    return -1
  else:
    return 1

def sort_phrase_idea_cmp(x, y):
  if x.start_time < y.start_time - 1e-2:
    return -1
  elif x.start_time > y.start_time + 1e-2:
    return 1
  elif 'PH' in x.id:
    return -1
  else:
    return 1

def clip_val(val, low, high):
  return max(low, min(high, val))