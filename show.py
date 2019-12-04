import numpy as np
import pandas as pd
from const import ACTION as A, STATE as S, REWARD as R
from first import zero_q, first_state
from pprint import pprint as pp
I = S.index
k = -1

def show_q(q, action):
  ur = np.unravel_index(np.arange(q.size), q.shape)
  valid = q[ur] != 0
  ur = tuple(idx[valid] for idx in ur)
  cols = {}
  for idx, s in enumerate(S.items):
    cols[s['name']] = [s['values'][i] for i in ur[idx]]
  cols[action.name] = (action.values[i] for i in ur[-1])
  cols['value'] = q[ur]
  df = pd.DataFrame(cols)

  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None
  ):
    print(df)

  return df


def test_show_q():
  q = zero_q(A.US_MOVE)
  q[tuple(int(i/2) for i in q.shape)] = 1
  show_q(q, A.US_MOVE)


def show_s(state):
  pp(
    {item['name']: item['values'][state[idx]]
    for idx, item in enumerate(S.items)})


def test_show_s():
  show_s(first_state())