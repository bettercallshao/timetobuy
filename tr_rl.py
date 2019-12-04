import numpy as np
from functools import lru_cache
from tqdm import tqdm
from const import STATE as S, ACTION as A, REWARD as R, FIGSIZE
from first import first_state, zero_q, random, zero_action
from show import show_q, show_s
import tr_model as m
I = S.index

ALPHA = 0.01
GAMMA = 0.3
EPSILON = 0.5
NITER = 100 * 1000
NRANDOM = 80 * 1000
NRESTART = 10


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

def train():
  """Run RL training loop"""
  ch_q = zero_q(A.CH_MOVE)
  us_q = zero_q(A.US_MOVE)

  for cnt in tqdm(range(NITER)):
    if cnt % NRESTART == 0:
      state = first_state()
    action = zero_action()
    action[A.index.US_MOVE] = random(A.US_MOVE)
    action[A.index.CH_MOVE] = random(A.CH_MOVE)

    rand_choice = cnt < NRANDOM or np.random.random(1)[0] < EPSILON

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

  np.save('us_q', us_q)
  np.save('ch_q', ch_q)
  return us_q, ch_q


@lru_cache()
def load_q():
  """Load saved Q and cache it"""
  return np.load('us_q.npy'), np.load('ch_q.npy')


def lock_extras(state):
  """Lock extra states to zero (which is what we trained on)"""
  first = first_state()
  new = np.copy(state)
  valid = {I.LAST_US_MOVE, I.LAST_CH_MOVE, I.USEC_GROWTH, I.CHEC_GROWTH}
  extras = list(set(range(S.N)) - valid)
  new[extras] = first[extras]
  return new

def trans(state):
  """Transit state based on greedy Q"""
  us_q, ch_q = load_q()
  action = zero_action()
  action[A.index.US_MOVE] = np.argmax(us_q[tuple(lock_extras(state))])
  state, r1 = m.trans(state, action, A.US_MOVE)
  action[A.index.CH_MOVE] = np.argmax(ch_q[tuple(lock_extras(state))])
  state, r2 = m.trans(state, action, A.CH_MOVE)
  return state


def test_train():
  train()
  us_q, ch_q = load_q()
  print('US Q')
  show_q(us_q, A.US_MOVE)
  print('CH Q')
  show_q(ch_q, A.CH_MOVE)
  state = first_state()
  show_s(state)
  for _ in range(5):
    state = trans(state)
    show_s(state)


def plot_ch_q():
  from pandasql import sqldf
  from matplotlib import pyplot as plt, rc

  us_q, ch_q = load_q()
  ch_df = show_q(ch_q, A.CH_MOVE)
  ch_df = ch_df.applymap(lambda x: {'LT5': 'LOW', 'B55': 'MEDIUM', 'B56': 'HIGH'}.get(x, x))
  sql = lambda q: sqldf(q, {'ch': ch_df})
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(*FIGSIZE)

  for move in ('DEESCALATE', 'ESCALATE', 'FX_ESCALATE'):
    sql(("SELECT USEC_GROWTH, CHEC_GROWTH, CH_MOVE, value AS {move} FROM ch WHERE " +
        "LAST_US_MOVE='ESCALATE' AND LAST_CH_MOVE='ESCALATE' AND USEC_GROWTH='LT2' AND CH_MOVE='{move}'")\
        .format(move=move))\
        .plot(x='CHEC_GROWTH', y=move, ax=ax, style='o-')
  plt.xlim((-0.2, 2.2))
  plt.xlabel('ch econ growth')
  plt.ylabel('value')
  plt.title('ch q (both escalate)')
  plt.grid(True)
  plt.savefig('ch_q.png')


def plot_usec_growth():
  from matplotlib import pyplot as plt, rc
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(*FIGSIZE)

  n = 4
  state = first_state()
  history = [state[I.USEC_GROWTH]]
  for _ in range(n-1):
    state = trans(state)
    history += [state[I.USEC_GROWTH]]

  plt.plot(range(n), history, 'o-')
  plt.yticks(range(S.USEC_GROWTH.N), [{'MT3': 'HIGH', 'B23': 'MEDIUM', 'LT2': 'LOW'}.get(a, a) for a in S.USEC_GROWTH.values])
  plt.ylabel('us econ growth')
  plt.xlabel('round')
  plt.title('us econ growth projection')
  plt.savefig('usec.png')


if __name__ == '__main__':
  train()
  plot_ch_q()
  plot_usec_growth()
