import numpy as np
from const import STATE as S, ACTION as A, REWARD as R, COEF as C
from first import first_state, zero_reward
from show import show_s
I = S.index

def dec(val, floor=0, d=1):
  return max(val - d, floor)


def inc(val, ceiling, d=1):
  return min(val + d, ceiling)


def is_resolution(state):
  return (state[I.LAST_US_MOVE] == S.LAST_US_MOVE.DEESCALATE and
          state[I.LAST_CH_MOVE] == S.LAST_CH_MOVE.DEESCALATE)


def adjust_growth(state):
  if is_resolution(state):
    state[I.USEC_GROWTH] = inc(state[I.USEC_GROWTH], S.USEC_GROWTH.MT3)
    state[I.CHEC_GROWTH] = inc(state[I.CHEC_GROWTH], S.CHEC_GROWTH.MT7, 2)
  else:
    state[I.USEC_GROWTH] = dec(state[I.USEC_GROWTH])
    if state[I.LAST_CH_MOVE] == S.LAST_CH_MOVE.FX_ESCALATE:
      state[I.CHEC_GROWTH] = dec(state[I.CHEC_GROWTH])
    else:
      state[I.CHEC_GROWTH] = dec(state[I.CHEC_GROWTH], d=2)
  return state


def trans(state, action, player):
  old = np.copy(state)
  m = {
    A.US_MOVE: I.LAST_US_MOVE,
    A.CH_MOVE: I.LAST_CH_MOVE
  }
  state[m[player]] = action[player.index]
  state = adjust_growth(state)

  reward = zero_reward()
  reward[R.index.US_PROFIT] = (0
    + (state[I.USEC_GROWTH] - old[I.USEC_GROWTH]) * C.USEC_GROWTH
    + (C.US_LEAD if (state[I.LAST_US_MOVE] == S.LAST_US_MOVE.ESCALATE) else 0)
  )
  reward[R.index.CH_PROFIT] = (0
    + (state[I.CHEC_GROWTH] - old[I.CHEC_GROWTH]) * C.CHEC_GROWTH
    + (C.CH_AUTO if (min(state[I.LAST_CH_MOVE], 1) == state[I.LAST_US_MOVE]) else 0)
    + (C.CH_FXCO if state[I.LAST_CH_MOVE] == S.LAST_CH_MOVE.FX_ESCALATE else 0)
  )
  return state, reward


def test_state_trans():
  state = first_state()
  show_s(state)
  state, r1 = trans(
    state, [A.US_MOVE.DEESCALATE, A.CH_MOVE.DEESCALATE], A.US_MOVE)
  show_s(state)
  print(r1)
  state, r2 = trans(
    state, [A.US_MOVE.DEESCALATE, A.CH_MOVE.DEESCALATE], A.CH_MOVE)
  show_s(state)
  print(r2)
