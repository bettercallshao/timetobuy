import numpy as np
from const import ACTION as A, STATE as S, REWARD as R
I = S.index

def first_state():
  s = np.zeros(S.N, dtype='int')
  s[I.LAST_US_MOVE] = S.LAST_US_MOVE.ESCALATE
  s[I.LAST_CH_MOVE] = S.LAST_CH_MOVE.ESCALATE
  s[I.USEC_GROWTH] = S.USEC_GROWTH.B23
  s[I.CHEC_GROWTH] = S.CHEC_GROWTH.MT7
  return s

def zero_policy():
  return np.zeros([len(item['values']) for item in S.items])

def zero_reward():
  return np.zeros(R.N)

def zero_action():
  return np.zeros(A.N, dtype='int')

def zero_q(action):
  return np.zeros([len(item['values']) for item in S.items] + [action.N])

def random(action):
  return np.random.randint(action.N)
