class Segment(object):
  def __init__(self, id, start_time, end_time, start_bar, end_bar):
    self.id = id
    self.start_time = start_time
    self.end_time = end_time
    self.start_bar = start_bar
    self.end_bar = end_bar
  
  def __repr__(self):
    return '[[ {} | {:.2f} ~ {:.2f} sec \t| Bar {}~{} ]]'.format(self.id, self.start_time, self.end_time, self.start_bar, self.end_bar)

class StructEvent(object):
  def __init__(self, id, start_time, end_time, start_barbeat, end_barbeat):
    self.id = id
    self.start_time = start_time
    self.end_time = end_time
    self.start_barbeat = start_barbeat
    self.end_barbeat = end_barbeat
  
  def __repr__(self):
    return '[[ {} | {:.2f} ~ {:.2f} sec \t| bar, beat {}~{} ]]'.format(self.id, self.start_time, self.end_time, self.start_barbeat, self.end_barbeat)

class NoteMCSV(object):
  def __init__(self, pitch, velocity, onset_sec, duration_sec):
    self.pitch = pitch
    self.velocity = velocity
    self.onset_sec = onset_sec
    self.duration_sec = duration_sec

  def __repr__(self):
    return '[Note MCSV] -- pitch: {} | velocity: {} | onset: {} | duration: {}'.format(
      self.pitch, self.velocity, self.onset_sec, self.duration_sec
    )

class ChordMCSV(object):
  def __init__(self, chord_type, bass, chord_notes, velocity, onset_sec, duration_sec):
    self.chord_type = chord_type
    self.bass = bass
    self.chord_notes = chord_notes
    self.velocity = velocity
    self.onset_sec = onset_sec
    self.duration_sec = duration_sec

  def __repr__(self):
    return '[Chord MCSV] -- type: {} | bass: {} | notes: {} | velocity: {} | onset: {} | duration: {}'.format(
      self.chord_type, self.bass, self.chord_notes, self.velocity, self.onset_sec, self.duration_sec
    )