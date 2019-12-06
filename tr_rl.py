import numpy as np
from tqdm import tqdm
from const import STATE as S, ACTION as A, REWARD as R
from first import first_state, zero_q, random, zero_action
from tr_model import trans
from show import show_q
I = S.index

ALPHA = 0.01
GAMMA = 0.3
EPSILON = 0.2

def q_update(state, next_state, q, a, r):
  q[tuple(state)][a] += (
    ALPHA * (
      r
      + GAMMA * np.max(q[tuple(next_state)])
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

    rand_choice = cnt < nrandom or np.random.random(1)[0] < EPSILON

    if not rand_choice:
      action[A.index.US_MOVE] = np.argmax(us_q[tuple(state)])
    next_state, reward = trans(state, action, A.US_MOVE)
    us_q = q_update(state, next_state, us_q,
      next_state[S.index.LAST_US_MOVE], reward[R.index.US_PROFIT])
    ch_q = q_update(state, next_state, ch_q,
      next_state[S.index.LAST_CH_MOVE], reward[R.index.CH_PROFIT])
    state = next_state

    if not rand_choice:
      action[A.index.CH_MOVE] = np.argmax(ch_q[tuple(state)])
    next_state, reward = trans(state, action, A.CH_MOVE)
    us_q = q_update(state, next_state, us_q,
      next_state[S.index.LAST_US_MOVE], reward[R.index.US_PROFIT])
    ch_q = q_update(state, next_state, ch_q,
      next_state[S.index.LAST_CH_MOVE], reward[R.index.CH_PROFIT])
    state = next_state

  return us_q, ch_q


def tr_train():
  ch_q = zero_q(A.CH_MOVE)
  us_q = zero_q(A.US_MOVE)

  us_q, ch_q = rl_both(us_q, ch_q, 300*1000, 10, 150*1000)
  print()
  print('US Q')
  show_q(us_q, A.US_MOVE)
  print()
  print('CH Q')
  show_q(ch_q, A.CH_MOVE)

  with open('us_q.npy', 'wb') as f:
    np.save(f, us_q)
  with open('ch_q.npy', 'wb') as f:
    np.save(f, ch_q)
