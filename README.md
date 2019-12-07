# timetobuy (aka trade war as a game)
Course project for reinforcement learning

Authors: Shihan Sun, Shaoqing Tan

## Code Breakdown

### Utilities

* const.py - Handles definition of constants.
* first.py - Handles initial states of variables.
* show.py - Handles printing states and Q map.
* run.py - Runs the code in order and produces plots.

### Trade War (tr)

* tr_model.py - Defines trade war model (state transition and reward).
* tr_rl.py - Runs Q-Learning for both players in the game.

### T-Bond Trading (pm)

* pm_model.py - Defines trading model (state transition and reward).
* pm_tr.py - Runs Q-Learning for most profit.

### Products

* Final Project 3547.pptx - Presentation
* us_q.npy, ch_q.npy - Q maps for trade war game.
* pm_q.npy - Q map for t-bond trading game.
* ch_q.png - A sample of player CH's Q map.
* usec.npy - The projection of us econ growth assuming both players are greedy.
* price.png - A random sample of price projections.
* mesh.png - A sample of trader's Q map.

## The Trade War Game

Transition / reward model is defined in `trans()` method in `tr_model.py`, Q learning is defined in `train()` method in `tr_rl.py`.

* ALPHA = 0.01
* GAMMA = 0.3
* EPSILON = 0.5

The model is relatively linear and simple, and there aren't many states. Therefore the traning process starts with completely random choices for maximum exploration, then switches to `epsilon = 0.5` greedy at the end for the both players to adjust to each others strategy.

Players are trained in an synchronous manner, they takes turns and rewards are attributed to both players last moves.

![yes](ch_q.png?raw=true)

![yes](usec.png?raw=true)

## The T-Bond Trading Game

Transition / reward model is defined in `trans()` method in `pm_model.py`, Q learning is defined in `train()` method in `pm_rl.py`.

* ALPHA = 0.01
* GAMMA = 0.98
* EPSILON = 0.5

This game assumes the trade war game to play out deterministically and a model of price is based on the us econ growth with random noise.

The game is episodic in nature, and the agent can buy and sell during four rounds of play and is forced to sell everything at the end. Gamma value is high here to accommodate the episodic nature of the game. The game is restarted per episode during training.

There are more states in this game therefore more iterations are necessary.

![yes](price.png?raw=true)

![yes](mesh.png?raw=true)
