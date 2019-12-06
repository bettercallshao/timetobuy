import numpy as np
import pandas as pd
from const import ACTION as A, STATE as S, REWARD as R
from first import zero_q, first_state
from pprint import pprint as pp
I = S.index
k = -1

def show_q(q, action):
  total = pd.DataFrame({k: [k]})
  for idx, item in enumerate(S.items):
    df = pd.DataFrame({
      item['name']: item['values'],
      idx: range(len(item['values']))
    })
    df[k] = k
    total = pd.merge(total, df, on=k)
  df = pd.DataFrame({
    action.name: action.values,
    S.N: range(action.N)
  })
  df[k] = k
  total = pd.merge(total, df, on=k)
  total['value'] = 0

  for idx, row in total.iterrows():
    total.loc[idx, 'value'] = q[tuple(row[i] for i in range(S.N+1))]

  total = total[[c for c in total.columns if isinstance(c, str)]]
  total = total[total['value'] != 0].reset_index(drop=True)

  with pd.option_context(
    'display.max_rows', None,
    'display.max_columns', None,
    'display.width', None
  ):
    print(total)


def test_show_q():
  show_q(zero_q(A.US_MOVE), A.US_MOVE)


def show_s(state):
  pp(
    {item['name']: item['values'][state[idx]]
    for idx, item in enumerate(S.items)})


def test_show_s():
  show_s(first_state())