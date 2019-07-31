# TODO add readme


## News
Combined cheng’s and sami’s latest
- reward is to be played with more
	- sum of rewards * max reward ?
- check if action/observation space does anything
	- didn't check this but I have no idea how gym doesn't blow up without the dedicated spaces
- done is not always False
- numpy arrays for states
- refactored the zillion different ways of keeping time steps
	- current_step = current training step (resets on every reset())
	- global_current_step = a better name for timesteps if that was its prupose. (if not I have no idea why it was there.)
	- T = another timestep cap on the episode (does the same work with 95 steps on the training - maybe.)


## Problems

Simulation is very black box.
Discuss (with civil engineering guys) how this simulation and real life works in general. What a good solution looks like. what opening valves change. Is this a complex calculus problem (i.e. flooding and depths are differential equations with respect to each other and orifices.)

(Right now model opens up the walves to the top. Doing that does minimize flood.)

rewarding max flood results in similar with -max flood (which is very suspicious)

what is success. Do we want no flood or flood < x

### Minor problems

Visualize=1 does not trigger render, so it's called from the environment.

## Note

Entire training process can be plotted, instead of just the final episode. Plotting does some more than necessary logging.
