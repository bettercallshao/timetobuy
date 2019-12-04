import numpy as np
from functools import lru_cache
from tqdm import tqdm
from const import STATE as S, ACTION as A, REWARD as R, FIGSIZE
from first import first_state, zero_q, random, zero_action
from show import show_q, show_s
import pm_model as pm
import tr_rl as tr
I = S.index
plot_price = pm.plot_price()

ALPHA = 0.01
GAMMA = 0.98
EPSILON = 0.5
NITER = 1000 * 1000
NRANDOM = 800 * 1000
NRESTART = 4


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
  q = zero_q(A.TRADE)

  for cnt in tqdm(range(NITER)):
    if cnt % NRESTART == 0:
      state = first_state()
      state = pm.trans_price(state)
    if cnt % NRESTART == NRESTART-1:
      end = True
    else:
      end = False
    action = zero_action()
    action[A.index.TRADE] = random(A.TRADE)

    rand_choice = cnt < NRANDOM or np.random.random(1)[0] < EPSILON

    if not rand_choice:
      action[A.index.TRADE] = np.argmax(q[tuple(state)])
    next_state, reward = pm.trans_holding(state, action, end=end)
    next_state = tr.trans(next_state)
    next_state = pm.trans_price(next_state)
    q = q_update(state, next_state, q, action[A.index.TRADE],
                 reward[R.index.TRADER_PROFIT])
    state = next_state

  np.save('pm_q', q)
  return q


def plot_mesh():
  from pandasql import sqldf
  from matplotlib import pyplot as plt, rc, cm
  from mpl_toolkits.mplot3d import Axes3D

  q = np.load('pm_q.npy')
  fig = plt.figure(figsize=FIGSIZE)
  ax = Axes3D(fig)
  bs = q[S.LAST_US_MOVE.DEESCALATE][S.LAST_CH_MOVE.DEESCALATE][S.USEC_GROWTH.MT3][S.CHEC_GROWTH.B67][:][:][:]
  xr = np.linspace(pm.price(S.PRICE.values[0]), pm.price(S.PRICE.values[-1]), bs.shape[0])
  yr = np.linspace(pm.price(S.HOLDING.values[0]), pm.price(S.HOLDING.values[-1]), bs.shape[1])
  xx, yy = np.meshgrid(yr, xr)

  yi = np.r_[range(0,40)]
  xi = np.r_[list(range(65, 104))]
  bs = np.max(bs, axis=2)

  xx = xx[65:100, [0,2,4,8,10]]
  yy = yy[65:100, [0,2,4,8,10]]
  zz = bs[65:100, [0,2,4,8,10]]

  ax.plot_surface(xx, yy, zz, cmap=plt.get_cmap('jet'), linewidth=0.1, antialiased=False)
  ax.set_ylabel('price')
  ax.set_xlabel('holding')
  ax.set_zlabel('value')
  plt.title('value map (both deescalate)')

  plt.savefig('mesh.png')


if __name__ == '__main__':
  train()
  plot_mesh()
