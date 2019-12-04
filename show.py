import numpy as np
import pandas as pd
from const import ACTION as A, STATE as S, REWARD as R
from first import zero_q
I = S.index

def show_q(q, action):
  total = pd.DataFrame({'k': ['k']})
  for idx, item in enumerate(S.items):
    df = pd.DataFrame({
      item['name']: item['values'],
      str(idx): range(len(item['values']))
    })
    df['k'] = 'k'
    total = pd.merge(total, df, on='k')
  a = A.items[]
  df = pd.DataFrame({
    A
  })
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(total)

def test_show_q():
  show_q(zero_q(A.US_MOVE))