import numpy as np
from tqdm import tqdm
from const import STATE as S, ACTION as A, REWARD as R
from first import first_state, zero_q, random, zero_action
from tr_model import trans
from show import show_q
I = S.index

ALPHA = 0.01
GAMMA = 0.3
EPSILON = 0.1

def q_update(state, next_state, q, a, r):
  q[tuple(state)][a] += (
    ALPHA * (
      r
      + GAMMA * np.max(us_q[tuple(next_state)])
      - q[tuple(state)][a]
    )
  )
  return q

def rl_both(us_q, ch_q, niter, nrestart, nrandom):
  for cnt in tqdm(range(niter)):
    if cnt % nrestart == 0:
      state = first_state()
    action = zero_action()
    action[A.index.US_MOVE] = random(A.US_MOVE)
    action[A.index.CH_MOVE] = random(A.CH_MOVE)

    next_state, reward = trans(state, action, A.US_MOVE)
    us_q = q_update(state, next_state, us_q,
      next_state[S.index.LAST_US_MOVE], reward[R.index.US_PROFIT])
    ch_q = q_update(state, next_state, ch_q,
      next_state[S.index.LAST_CH_MOVE], reward[R.index.CH_PROFIT])
    state = next_state

    next_state, reward = trans(state, action, A.CH_MOVE)
    us_q = q_update(state, next_state, us_q,
      next_state[S.index.LAST_US_MOVE], reward[R.index.US_PROFIT])
    ch_q = q_update(state, next_state, ch_q,
      next_state[S.index.LAST_CH_MOVE], reward[R.index.CH_PROFIT])
    state = next_state

  return us_q, ch_q

ch_q = zero_q(A.CH_MOVE)
us_q = zero_q(A.US_MOVE)

us_q, ch_q = rl_random_both(us_q, ch_q, 299999, 20)
print()
print('US Q')
show_q(us_q, A.US_MOVE)
print()
print('CH Q')
show_q(ch_q, A.CH_MOVE)
