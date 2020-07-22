# create  key_map and chord_profile

from ast import literal_eval
import pickle


chord_profile = {}

# map for different names for same key
key_map = {
  'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 
  'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11
}

profile_file = './chord_profile.txt' # written by hand (Sean Wu)

if __name__ == '__main__':
  with open(profile_file, 'r', encoding='utf-8') as f:
    for l in f.readlines():
      l = l.strip().split()
      ch_type, ch_profile = l[2], ' '.join(l[3:])
      if ch_type == '\'\'':
        ch_type = ''
      
      # replace (empty) to '' for futher usage
      if(ch_type=='(empty)'):
        ch_type=''
    
      print (ch_type, ch_profile)

      # literal_eval() turn string to list
      chord_profile[ ch_type ] = literal_eval(ch_profile)


  # for k, v in chord_profile.items():
  #   print (k, ':', v)
  # print (len(chord_profile))

  pickle.dump(chord_profile, open('../pickles/chord_profile.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
  pickle.dump(key_map, open('../pickles/key_map.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)