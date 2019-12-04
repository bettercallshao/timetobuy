import numpy as np
from const import STATE as S, ACTION as A, REWARD as R, COEF as C
I = S.index

def zero_policy():
  return np.zeros([len(item['values']) for item in S.items])
