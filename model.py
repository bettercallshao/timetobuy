import numpy as np
from const import STATE as S, ACTION as A, COEF as C
I = S.index

def first_state():
  s = np.zeros(S.items)
  s[I.LAST_US_MOVE] = S.LAST_US_MOVE.ESCALATE
  s[I.LAST_CH_MOVE] = S.LAST_CH_MOVE.ESCALATE

def _internal_trans(state):
  is_war = state[I.LAST_US_MOVE] + state[I.LAST_CH_MOVE] == 0
  if is_war:
    state[I.USEC_GROWTH] = max(0, state[I.USEC_GROWTH] - 1)
  else:
    state[I.USEC_GROWTH] = min(S.USEC_GROWTH.MT3, state[I.USEC_GROWTH] + 1)
  return state

def us_trans(state, action):
  state[I.LAST_US_MOVE] = action
  state = _internal_trans(state)
  return state, state[I.USEC_GROWTH] * C.USEC_GROWTH

def ch_trans(state, action):
  state[I.LAST_CH_MOVE] = action
  state = _internal_trans(state)
  return state, state[I.CHEC_GROWTH] * C.CHEC_GROWTH
