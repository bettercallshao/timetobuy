import yaml
import pprint

class O(object):
  def __repr__(self):
    d = {key: val for key, val in self.__dict__.items() if key[0] != '_'}
    return pprint.pformat(d, indent=2)

def objectize(items):
  root = O()
  setattr(root, 'items', items)
  setattr(root, 'index', O())
  setattr(root, 'N', len(items))
  for itemIdx, item in enumerate(items):
    setattr(root.index, item['name'], itemIdx)
    if 'values' in item:
      attr = O()
      setattr(attr, 'N', len(item['values']))
      setattr(attr, 'name', item['name'])
      setattr(attr, 'values', item['values'])
      setattr(attr, 'index', itemIdx)
      for idx, value in enumerate(item['values']):
        setattr(attr, value, idx)
      setattr(root, item['name'], attr)
  return root


def fill_range(l):
  for a in l:
    if 'range' in a:
      r = a['range']
      del a['range']
      a['values'] = [('P' if v >= 0 else 'N') + '%02d' % (abs(v)) for v in range(r['from'], r['to']+1)]


CONST = yaml.safe_load(open('constants.yaml'))
fill_range(CONST['STATE'])
STATE = objectize(CONST['STATE'])
ACTION = objectize(CONST['ACTION'])
REWARD = objectize(CONST['REWARD'])
COEF = O()
COEF.__dict__.update(CONST['COEF'])
FIGSIZE = [8, 6]