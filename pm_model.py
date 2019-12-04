import numpy as np
from const import STATE as S, ACTION as A, REWARD as R, COEF as C, FIGSIZE
from first import first_state, zero_reward, zero_action
from show import show_s
from tr_model import is_resolution
import tr_rl as tr
I = S.index


def price(state):
  text = state if isinstance(state, str) else S.PRICE.values[state[I.PRICE]]
  return int(text[1:]) / 10.0 * (-1 if text[0] == 'N' else 1)


def holding(state):
  text = state if isinstance(state, str) else S.HOLDING.values[state[I.HOLDING]]
  return int(text[1:]) * (-1 if text[0] == 'N' else 1)


def trans_price(state):
  old = np.copy(state)
  new = np.copy(state)

  is_cut = old[I.USEC_GROWTH] == S.USEC_GROWTH.LT2
  is_res = is_resolution(old)

  if is_cut and is_res:
    pair = (1, 1.5)
  elif is_cut and not is_res:
    pair = (-1.0, 0.5)
  elif not is_cut and is_res:
    pair = (0, 1)
  elif not is_cut and not is_res:
    pair = (-1.5, -1.0)

  diff = np.random.randint(pair[0] * 10, pair[1] * 10 + 1)
  new[I.PRICE] = old[I.PRICE] + diff

  # print('cut: {}, res: {}, pair: {}, diff: {}'.format(is_cut, is_res, pair, diff))
  return new


def trans_holding(state, action, end=False):
  old = np.copy(state)
  new = np.copy(state)
  reward = zero_reward()

  if end:
    new[I.HOLDING] = 0
    reward[R.index.TRADER_PROFIT] = price(old) * holding(old)
  elif action[A.index.TRADE] == A.TRADE.BUY:
    new[I.HOLDING] += 1
    reward[R.index.TRADER_PROFIT] = -price(old)
  else:
    new[I.HOLDING] -= 1
    reward[R.index.TRADER_PROFIT] = price(old)

  return new, reward


def plot_price():
  from matplotlib import pyplot as plt, rc
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(*FIGSIZE)

  for _ in range(6):
    history = []
    state = first_state()
    action = zero_action()
    state = trans_price(state)
    history = [price(state)]
    n = 4
    for _ in range(n-1):
      state = tr.trans(state)
      state = trans_price(state)
      history.append(price(state))
    plt.plot(range(n), history, '-')

  plt.ylabel('price')
  plt.xlabel('round')
  plt.title('price projection')
  plt.savefig('price.png')


def test_trans():
  state = first_state()
  action = zero_action()
  action[A.index.TRADE] = A.TRADE.BUY
  for _ in range(3):
    state = tr.trans(state)
    state = trans_price(state)
    state, reward = trans_holding(state, action)
    show_s(state)
    print(reward)
  state = tr.trans(state)
  state = trans_price(state)
  state, reward = trans_holding(state, action, end=True)
  show_s(state)
  print(reward)
