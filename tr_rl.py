import numpy as np
from functools import lru_cache
from tqdm import tqdm
from const import STATE as S, ACTION as A, REWARD as R
from first import first_state, zero_q, random, zero_action
from show import show_q, show_s
import tr_model as m
I = S.index

ALPHA = 0.01
GAMMA = 0.3
EPSILON = 0.2


def q_update(state, next_state, q, a, r):
  """Update Q map from action and reward"""
  q[tuple(state)][a] += (
    ALPHA * (
      r
      + GAMMA * np.max(q[tuple(next_state)])
      - q[tuple(state)][a]
    )
  )
  return q

def rl_both(us_q, ch_q, niter, nrestart, nrandom):
  """Run RL training loop"""
  for cnt in tqdm(range(niter)):
    if cnt % nrestart == 0:
      state = first_state()
    action = zero_action()
    action[A.index.US_MOVE] = random(A.US_MOVE)
    action[A.index.CH_MOVE] = random(A.CH_MOVE)

    rand_choice = cnt < nrandom or np.random.random(1)[0] < EPSILON

    if not rand_choice:
      action[A.index.US_MOVE] = np.argmax(us_q[tuple(state)])
    next_state, reward = m.trans(state, action, A.US_MOVE)
    us_q = q_update(state, next_state, us_q,
      next_state[S.index.LAST_US_MOVE], reward[R.index.US_PROFIT])
    ch_q = q_update(state, next_state, ch_q,
      next_state[S.index.LAST_CH_MOVE], reward[R.index.CH_PROFIT])
    state = next_state

    if not rand_choice:
      action[A.index.CH_MOVE] = np.argmax(ch_q[tuple(state)])
    next_state, reward = m.trans(state, action, A.CH_MOVE)
    us_q = q_update(state, next_state, us_q,
      next_state[S.index.LAST_US_MOVE], reward[R.index.US_PROFIT])
    ch_q = q_update(state, next_state, ch_q,
      next_state[S.index.LAST_CH_MOVE], reward[R.index.CH_PROFIT])
    state = next_state

  return us_q, ch_q


def train():
  """Wraps training process and saves parameters"""
  ch_q = zero_q(A.CH_MOVE)
  us_q = zero_q(A.US_MOVE)

  us_q, ch_q = rl_both(us_q, ch_q, 30*1000, 10, 15*1000)
  print()
  print('US Q')
  show_q(us_q, A.US_MOVE)
  print()
  print('CH Q')
  show_q(ch_q, A.CH_MOVE)

  np.save('us_q', us_q)
  np.save('ch_q', ch_q)


@lru_cache()
def load_q():
  """Load saved Q and cache it"""
  return np.load('us_q.npy'), np.load('ch_q.npy')


def trans(state):
  """Transit state based on greedy Q"""
  us_q, ch_q = load_q()
  action = zero_action()
  action[A.index.US_MOVE] = np.argmax(us_q[tuple(state)])
  print(us_q[tuple(state)])
  state, r1 = m.trans(state, action, A.US_MOVE)
  action[A.index.CH_MOVE] = np.argmax(ch_q[tuple(state)])
  print(ch_q[tuple(state)])
  state, r2 = m.trans(state, action, A.CH_MOVE)
  return state


def test_train():
  train()
  state = first_state()
  show_s(state)
  for _ in range(4):
    state = trans(state)
    show_s(state)


if __name__ == '__main__':
  test_train()
