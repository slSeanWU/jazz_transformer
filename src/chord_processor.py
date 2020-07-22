import pickle
import numpy as np

class ChordProcessor(object):
  def __init__(self, ch_profile_file, key_map_file, bass_base=36, chord_base=48):
    # bass_base=36 : C2
    # chord_base=48 : C3
    self.chord_profile = pickle.load(open(ch_profile_file, 'rb'))
    self.key_map = pickle.load(open(key_map_file, 'rb'))
    self.bass_base = bass_base # the anchor for bass note
    self.chord_base = chord_base # the anchor for list of chord notes

  # parse chord_literal to (tone,type,slash)
  def _parse_chord_literal(self, ch):
    if len(ch) < 2:
      # tone inside [C,D,E,F,G,A,B]
      tone, ch_type, slash = ch[0], '', ''
    elif ch[1] in ['#', 'b']:
      # tone contains #,b
      tone, ch_type = ch[:2], ch[2:].split('/')
    else:
      tone, ch_type = ch[:1], ch[1:].split('/')

    if not ch_type:
      # triad with no slash
      ch_type, slash = '', ''
    elif len(ch_type) > 1:
      ch_type, slash = ch_type[0], ch_type[1]
    else:
      ch_type, slash = ch_type[0], ''

    # print ('[[parsed]] {} --> {} | {} | {}'.format(ch, tone, ch_type, slash))
    return tone, ch_type, slash

  # convert chord literal to list of notes 
  def compute_notes(self, chord_literal, require_parsing=True, tone=None, ch_type=None, slash=None):
    if require_parsing:
      tone, ch_type, slash = self._parse_chord_literal(chord_literal)

    if not slash:
      slash = tone

    chord = list( np.asarray( self.chord_profile[ ch_type ] ) + self.chord_base + self.key_map[tone] )
    bass = self.bass_base + self.key_map[slash]

    # print ('(result) chord: {}, bass: {}'.format(chord, bass))
    # print ('-------------------------------------------------')
    return chord, bass

if __name__ == '__main__':
  ch_proc = ChordProcessor('../pickles/chord_profile.pkl', '../pickles/key_map.pkl')
  pickle.dump(ch_proc, open('../pickles/chord_processor.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)