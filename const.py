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
  for itemIdx, item in enumerate(items):
    setattr(root.index, item['name'], itemIdx)
    attr = O()
    for idx, value in enumerate(item['values']):
      setattr(attr, value, idx)
    setattr(root, item['name'], attr)
  return root

CONST = yaml.safe_load(open('constants.yaml'))
STATE = objectize(CONST['STATE'])
ACTION = objectize(CONST['ACTION'])
COEF = O()
COEF.__dict__.update(CONST['COEF'])