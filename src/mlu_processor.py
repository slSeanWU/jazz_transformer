import re
import pickle

class MLUProcessor(object):
  def __init__(self, max_back_ref=8):
    self.mlu_types = set()
    self.mlu_subtypes = {}
    self.rep_vars = set()
    self.max_back_ref = max_back_ref

  def __repr__(self):
    return 'type: {}\nsubtype: {}\nrep variations: {}'.format(self.mlu_types, self.mlu_subtypes, self.rep_vars)

  def parse_mlu_literal(self, raw_mlu, build_vocab=False):
    raw_mlu = raw_mlu.replace('~', '').replace('*', '')
    raw_mlu = re.sub(':.+', '', raw_mlu)

    if 'void->' in raw_mlu:
      has_void = True
      raw_mlu = raw_mlu.replace('void->', '')
    else:
      has_void = False

    rep = re.search(r'(#+)(\d*)([=\-\+]?)', raw_mlu)
    if rep:
      rep_variation = rep.group(3) if rep.group(3) is not None else ''

      if rep.group(2) and int(rep.group(2)) > self.max_back_ref:
        rep_backref, rep_variation = '', ''
      elif rep.group(2) and int(rep.group(2)) <= self.max_back_ref:
        rep_backref = int(rep.group(2))
      else:
        rep_backref = len(rep.group(1))
    else:
      rep_backref, rep_variation = '', ''

    typ = re.search(r'([a-z]+)((_.*)?)', raw_mlu).group(0).split('_')
    typ, sub_typ = typ[0], typ[1] if len(typ) > 1 else ''
    
    if typ == 'oscillation':
      typ, sub_typ = 'rhythm', 'mr'
    if typ == 'quote':
      typ = 'melody'

    if build_vocab:
      self.rep_vars |= set([rep_variation])
      self.mlu_types |= set([typ])
      if typ not in self.mlu_subtypes:
        self.mlu_subtypes[typ] = set([sub_typ])
      else:
        self.mlu_subtypes[typ] |= set([sub_typ])

    return has_void, rep_backref, rep_variation, typ, sub_typ

if __name__ == '__main__':
  with open('mlus_events.txt', 'r') as f:
    lines = f.readlines()
  
  lines = [l.strip().split()[0] for l in lines]
  # print (lines)

  # has_void = False
  # repetition = ''
  # for mlu in lines:
  #   if 'void->' in mlu:
  #     has_void = True
  #     mlu = mlu.replace('void->', '')
    
  #   # if '#' in mlu:
  #   #   rep_part = re.search(r'(#+)(\d*)([=\-\+]?)', mlu)
  #   #   print ('{:20} --> {:10} | {:10} | {:10}'.format(mlu, rep_part.group(1), rep_part.group(2), rep_part.group(3) ))

  #   idea_part = re.search(r'([a-z]+)((_.*)?)', mlu)
  #   print ('{:20} --> {}'.format(mlu, idea_part.group(0).split('_')))

  proc = MLUProcessor()
  for mlu in lines:
    has_void, rep_backref, rep_variation, typ, sub_typ = proc.parse_mlu_literal(mlu, build_vocab=True)
    print ('{:20} --> {:6} | {:1} | {:1} | {:10} | {:10}'.format(mlu, has_void, rep_backref, rep_variation, typ, sub_typ))

  print (proc)
  pickle.dump(proc, open('../pickles/mlu_processor.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)