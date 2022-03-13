#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

import shutil
import os
import time
from collections import deque
from copy import deepcopy
import torch

from utils import OUNoise, make_gif, empty_torch_queue
from utils import Logger
from utils import create_env_wrapper

class Agent(object):

    def __init__(self, config, policy, global_episode, n_agent=0, agent_type='exploration', log_dir=''):
        """
        agent_type: exploration: + Noise
        agent_type: exploitation: without Noise
        """
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir

        # create_environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"])
        self.ou_noise.reset()

        self.actor = policy
        print("Agent ", n_agent, self.actor.device)

        # Logger
        log_path = f"{log_dir}/agent-{agent_type}-{n_agent}"
        self.logger = Logger(log_path)

    def update_actor_learner(self, learner_w_queue, training_on):
        """Update local actor to the actor from learner. """
        """[Learner Worker]--(policy weights)-->[Exploration Worker]*(n_agents)"""
        if not training_on.value:
            return
        try:
            source = learner_w_queue.get_nowait()
        except:
            return
        target = self.actor
        for target_param, source_param in zip(target.parameters(), source):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []

        while training_on.value:
            episode_reward = 0
            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.agent_type == "exploration":
                if self.local_episode % 100 == 0:
                    print("-"*100)
                    print(f"Exploration Agent: {self.n_agent}  episode {self.local_episode}")
                    print("-" * 100)
            else:
                if self.local_episode % 100 == 0:
                    print(f"Exploitation Agent: {self.n_agent}  episode {self.local_episode}")

            ep_start_time = time.time()
            # env reset
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            done = False

            while not done:
                # print(type(state)) debug
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    action = self.ou_noise.get_action(action,num_steps)
                    action = action.squeeze(0)
                else: # exploitation
                    action = action.detach().cpu().numpy().flatten()

                next_state, reward, done = self.env_wrapper.step(action)
                num_steps += 1

                if num_steps == self.max_steps:
                    done = False
                episode_reward += reward

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

                # exp_buffer，用于计算TD(N) Target
                self.exp_buffer.append((state, action, reward))

                """TD(N-step)"""
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft() # queue 先入先出
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']

                    for (_,_,r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate'] # decay ?

                    if self.agent_type == "exploration":
                        try:
                            replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                        except:
                            pass

                """State Update"""
                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                        if self.agent_type == "exploration":
                            try:
                                replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                            except:
                               pass
                    break

            # Log metrics
            step = update_step.value
            self.logger.scalar_summary(f"agent_{self.agent_type}/reward", episode_reward, step)
            # print(self.global_episode)
            # print(self.local_episode)
            # self.logger.scalar_summary(f"agent_{self.agent_type}/reward_episode", episode_reward, self.global_episode.value)
            self.logger.scalar_summary(f"agent_{self.agent_type}/reward_episode2", episode_reward, self.local_episode)
            self.logger.scalar_summary(f"agent_{self.agent_type}/episode_timing", time.time() - ep_start_time, step)

            # Saving
            # Saving agent
            reward_outperformed = episode_reward - best_reward > self.config["save_reward_threshold"]
            time_to_save = self.local_episode % self.num_episode_save == 0
            if self.agent_type == "exploitation" and (time_to_save or reward_outperformed):
                if episode_reward > best_reward:
                    best_reward = episode_reward
                # only save best model
                self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")

            rewards.append(episode_reward)
            if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                # update model weight
                self.update_actor_learner(learner_w_queue, training_on)

        empty_torch_queue(replay_queue)
        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self, output_dir_name):
        import matplotlib.pyplot as plt

        dir_name = output_dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy()
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")