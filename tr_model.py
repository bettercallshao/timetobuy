import numpy as np
from const import STATE as S, ACTION as A, REWARD as R, COEF as C
from first import first_state, zero_reward
I = S.index

def trans(state, action):
  state[I.LAST_US_MOVE] = action[A.index.US_MOVE]
  state[I.LAST_CH_MOVE] = action[A.index.CH_MOVE]

  def dec(val, floor=0):
    return max(val - 1, floor)
  def inc(val, ceiling):
    return min(val + 1, ceiling)

  is_war = state[I.LAST_US_MOVE] + state[I.LAST_CH_MOVE] == 0
  if is_war:
    state[I.USEC_GROWTH] = dec(state[I.USEC_GROWTH])
    state[I.CHEC_GROWTH] = dec(state[I.CHEC_GROWTH])
  else:
    state[I.USEC_GROWTH] = inc(state[I.USEC_GROWTH], S.USEC_GROWTH.MT3)
    state[I.CHEC_GROWTH] = inc(state[I.CHEC_GROWTH], S.CHEC_GROWTH.MT6)

  reward = zero_reward()
  reward[R.index.US_PROFIT] = state[I.USEC_GROWTH] * C.USEC_GROWTH
  reward[R.index.CH_PROFIT] = state[I.CHEC_GROWTH] * C.CHEC_GROWTH
  return state, reward

def test_state_trans():
  state = first_state()
  state, r1 = trans(
    state, [A.US_MOVE.DEESCALATE, A.CH_MOVE.DEESCALATE])
  state, r2 = trans(
    state, [A.US_MOVE.ESCALATE, A.CH_MOVE.ESCALATE])

  assert r1[0] > r2[0]
  assert r1[1] > r2[1]
  assert r1[2] == r2[2]
