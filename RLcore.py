# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer
)


class Agent(object):
    def __init__(self, processor=None):
        self.processor = processor
        self.training = False
        self.step = 0

    def get_config(self):
        return {}

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):
       # env.close()
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observation = None
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                
                #nb_max_episode_steps = env.num_steps-1

                if observation is None:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observation = deepcopy(env.reset())
                    
                    nb_max_episode_steps = (env.num_steps - 1)

                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    assert observation is not None

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            action = env.action_space.sample()
                        else:
                            action = start_step_policy(observation)
                        if self.processor is not None:
                            action = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observation, reward, done  = env.step(action)
                        #print("action")
                        observation = deepcopy(observation)
                        if self.processor is not None:
                            observation, reward, done  = self.processor.process_step(observation, reward, done )
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                            observation = deepcopy(env.reset())
                            nb_max_episode_steps = (env.num_steps - 1)

                            if self.processor is not None:
                                observation = self.processor.process_observation(observation)
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observation is not None

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = np.float32(0)
               # accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, done  = env.step(action)
                    #print(observation,r,done)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, done  = self.processor.process_step(observation, r, done )
                   # for key, value in info.items():
                    #    if not np.isreal(value):
                     #       continue
                      #  if key not in accumulated_info:
                        #    accumulated_info[key] = np.zeros_like(value)
                       # accumulated_info[key] += value
                    callbacks.on_action_end(action)

                   # print(r)
                   
                    reward += r
                    if done:
                        break
                
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True
                metrics = self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'metrics': metrics,
                    'episode': episode,
                    'date': env.date,
                }
               # print(episode_reward)
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    self.forward(observation)
                    self.backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }

                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observation = None
                    episode_step = None
                    episode_reward = None
                    env.sim.report()
                    env.sim.close()       
        
                    self.step += 1
                    
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()
        #env.graph("plots/test_plot_")
        return history 
    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1, plt=""):
       
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. Please call `compile()` before `test()`.')
        if action_repetition < 1:
            raise ValueError('action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0.
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset(test=True))
            nb_max_episode_steps = (env.num_steps - 1)
            
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done  = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done  = self.processor.process_step(observation, r, done )
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    nb_max_episode_steps = (env.num_steps - 1)
                    if self.processor is not None:
                        observation = self.processor.process_observation(observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
               # accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d  = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d  = self.processor.process_step(observation, r, d )
                    callbacks.on_action_end(action)
                    reward += r
                    #for key, value in info.items():
                    #    if not np.isreal(value):
                     #       continue
                     #   if key not in accumulated_info:
                      #      accumulated_info[key] = np.zeros_like(value)
                      #  accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward
               # print(episode_reward)
                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'date': env.date,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1
                
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
                'date': env.date,
            }
    
            callbacks.on_episode_end(episode, env, episode_logs) #help
            if(episode != nb_episodes - 1) and (plt != ""):
                env.graph(plt)
        if(plt != ""):
            env.graph(plt, end=True)
        callbacks.on_train_end()
        self._on_test_end()
        return history

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    @property
    def metrics_names(self):
        return []

    def _on_train_begin(self):
        pass

    def _on_train_end(self):
        pass

    def _on_test_begin(self):
        pass

    def _on_test_end(self):
        pass