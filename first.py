import numpy as np
from const import STATE as S, REWARD as R
I = S.index

def first_state():
  s = np.zeros(len(S.items))
  s[I.LAST_US_MOVE] = S.LAST_US_MOVE.ESCALATE
  s[I.LAST_CH_MOVE] = S.LAST_CH_MOVE.ESCALATE
  s[I.USEC_GROWTH] = S.USEC_GROWTH.B23
  s[I.CHEC_GROWTH] = S.CHEC_GROWTH.LT6
  return s

def zero_policy():
  return np.zeros([len(item['values']) for item in S.items])

def zero_reward():
  return np.zeros(len(R.items))
