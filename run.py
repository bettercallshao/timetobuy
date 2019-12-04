import tr_rl as tr
import pm_rl as pm

if __name__ == '__main__':
  tr.train()
  tr.plot_ch_q()
  tr.plot_usec_growth()
  pm.train()
  pm.plot_price()
  pm.plot_mesh()
