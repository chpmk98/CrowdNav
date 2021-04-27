import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import random
import pysocialforce as psf
from matplotlib import patches
from numpy.linalg import norm
from scipy.stats import poisson
from scipy.spatial import ConvexHull
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.group import Group
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist
from crowd_sim.envs.utils.utils import dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        # environment configuration
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.groups = None
        self.group_objs = None
        self.global_time = None
        self.human_times = None

        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.disccomfort_dist = None
        self.discomfort_penalty_factor = None

        self.progress_reward = None
        self.success_reward = None
        self.discomfort_penalty = None
        self.collision_penalty = None
        self.group_penalty = None
        self.collision_dist = None
        self.discomfort_dist = None

        # simulation configuration
        self.config = None
        self.enable_psf = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.group_num = None
        self.one_group = None

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

    def configure(self, config):
        self.config = config

        # environment configuration
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.enable_psf = config.getboolean('humans', 'enable_psf')

        # reward function
        self.success_reward = config.getfloat('old_reward', 'success_reward')
        self.collision_penalty = config.getfloat('old_reward', 'collision_penalty')
        self.disccomfort_dist = config.getfloat('old_reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('old_reward', 'discomfort_penalty_factor')

        self.progress_reward = config.getfloat('reward', 'progress_reward')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.discomfort_penalty = config.getfloat('reward', 'discomfort_penalty')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.group_penalty = config.getfloat('reward', 'group_penalty')
        self.collision_dist = config.getfloat('reward', 'collision_dist')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')

        # simulation configuration
        if config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.square_width = config.getfloat('sim', 'square_width')
            self.human_num = config.getint('sim', 'human_num')
            self.one_group = config.getboolean('sim', 'one_group')

        else:
            raise NotImplementedError

        # logging
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_square_crossing_human())
        elif rule == 'circle_crossing':
            self.humans = []
            for i in range(human_num):
                self.humans.append(self.generate_circle_crossing_human())
        elif rule == 'group_circle_crossing':
            self.group_num = poisson.rvs(1.2)
            if (self.group_num <= 0) or (self.group_num > self.human_num):
                self.group_num = self.human_num
            self.humans = []
            self.group_objs = []
            self.groups = []
            for i in range(self.group_num):
                self.group_objs.append(self.generate_circle_crossing_group())
                self.groups.append([])
            for i in range(human_num):
                # pick a group for the human to be in
                group_ind = np.random.randint(self.group_num)
                self.groups[group_ind].append(i)
                self.humans.append(self.generate_grouped_human(self.group_objs[group_ind]))
        elif rule == 'mixed':
            # mix different raining simulation with certain distribution
            static_human_num = {0: 0.05, 1: 0.2, 2: 0.2, 3: 0.3, 4: 0.1, 5: 0.15}
            dynamic_human_num = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.1}
            static = True if np.random.random() < 0.2 else False
            prob = np.random.random()
            for key, value in sorted(static_human_num.items() if static else dynamic_human_num.items()):
                if prob - value <= 0:
                    human_num = key
                    break
                else:
                    prob -= value
            self.human_num = human_num
            self.humans = []
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                if human_num == 0:
                    human = Human(self.config, 'humans')
                    human.set(0, -10, 0, -10, 0, 0, 0)
                    self.humans.append(human)
                for i in range(human_num):
                    human = Human(self.config, 'humans')
                    if np.random.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = np.random.random() * width * 0.5 * sign
                        py = (np.random.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans:
                            if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.disc_ped_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    human.set(px, py, px, py, 0, 0, 0)
                    self.humans.append(human)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(human_num):
                    if i < 2:
                        human = self.generate_circle_crossing_human()
                    else:
                        human = self.generate_square_crossing_human()
                    self.humans.append(human)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.disc_ped_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.disc_ped_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.disc_ped_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_circle_crossing_group(self):
        groupBoi = Group(self.config, 'groups')
        if self.randomize_attributes:
            groupBoi.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with group
            px_noise = (np.random.random() - 0.5) * groupBoi.v_pref
            py_noise = (np.random.random() - 0.5) * groupBoi.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.group_objs:
                min_dist = groupBoi.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        groupBoi.set(px, py, -px, -py, 0, 0, 0)
        return groupBoi

    def generate_grouped_human(self, group):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # assume a gaussian distribution of people within a group
            noise = np.random.normal(scale=group.stdev)
            px = group.px + noise * np.cos(angle)
            py = group.py + noise * np.sin(angle)
            gx = group.gx + noise * np.cos(angle)
            gy = group.gy + noise * np.sin(angle)
            collide = False
            for agent in [self.robot] + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'

        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
                else:
                    self.generate_random_human_position(human_num=self.human_num, rule=self.test_sim)
                # case_counter is always between 0 and case_size[phase]
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # for debugging purposes
                    self.human_num = 3
                    self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                    self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                    self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                    self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
                else:
                    raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        # initiate pysocialforce simulator
        if self.enable_psf:
            if self.robot.visible:
                initial_state = np.zeros((self.human_num+1, 6))
                for i, human in enumerate(self.humans):
                  initial_state[i, :] = np.array([human.px, human.py, human.vx+0.1, human.vy+0.1, human.gx, human.gy])
                initial_state[self.human_num, :] = np.array([self.robot.px, self.robot.py, self.robot.vx+0.1, self.robot.vy+0.1, self.robot.gx, self.robot.gy])
                groups = self.groups.append([self.human_num])
            else:
                initial_state = np.zeros((self.human_num, 6))
                for i, human in enumerate(self.humans):
                  initial_state[i, :] = np.array([human.px, human.py, human.vx+0.1, human.vy+0.1, human.gx, human.gy])
                groups = self.groups
            self.psf_sim = psf.Simulator(
                state=initial_state,
                groups=groups,
                obstacles=None,
                config_file="../pysocialforce/config/default.toml"
            )

        # initialize distance of robot to goal
        self.dgoal = [dist(self.robot.gx, self.robot.gy, self.robot.px, self.robot.py)]

        return ob

    def step(self, action):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        # store state, action value and attention weights
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values.append(self.robot.policy.action_values)
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights.append(self.robot.policy.get_attention_weights())

        # update robot
        self.robot.step(action)

        # update human: pysocialforce
        if self.enable_psf:
            self.psf_sim.step(3)
            ped_states, group_states = self.psf_sim.get_states()
            for i in range(self.human_num):
                [px, py, vx, vy, gx, gy, tau] = ped_states[-1, i, :]
                self.humans[i].set_position([px, py])
                self.humans[i].set_velocity([vx, vy])
            if self.robot.visible:
                self.psf_sim.peds.state[self.human_num, :] = np.array([self.robot.px,
                self.robot.py, self.robot.vx, self.robot.vy, self.robot.gx, self.robot.gy, 0.5])

        # update human: orca
        else:
            human_actions = []
            for human in self.humans:
                # observation for humans is always coordinates
                ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
                if self.robot.visible:
                    ob += [self.robot.get_observable_state()]
                human_actions.append(human.act(ob))
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)

        # update global time and record the first time the human reaches the goal
        self.global_time += self.time_step
        for i, human in enumerate(self.humans):
            if self.human_times[i] == 0 and human.reached_destination():
                self.human_times[i] = self.global_time

        # compute the observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        # compute distance from robot to pedestrians
        dped = np.zeros(self.human_num)
        for i, human in enumerate(self.humans):
            dped[i] = dist(human.px, human.py, self.robot.px, self.robot.py)

        # detect pedetrian collisions and discomfort
        coll_ped = np.array([1 if (dped[i] < self.collision_dist) else 0 for i in range(self.human_num)])
        disc_ped = np.array([1 if (dped[i] >= self.collision_dist) and (dped[i] < self.discomfort_dist) else 0 for i in range(self.human_num)])
        collision = any(i==1 for i in coll_ped)

        # compute distance from robot to goal
        self.dgoal.append(dist(self.robot.gx, self.robot.gy, self.robot.px, self.robot.py))

        # check if robot reached goal
        reached_goal = 1 if (self.dgoal[-1] < self.collision_dist) else 0

        # compute distance from robot to convex hull of groups
        dgrp = np.zeros(self.group_num)
        for j in range(self.group_num):
            # not a group; don't care about violating group discomfort
            if len(self.groups[j]) == 0 or len(self.groups[j]) == 1:
                dgrp[j] = self.discomfort_dist
            elif len(self.groups[j]) == 2:
                human_idx = self.groups[j]
                ped_states, _ = self.psf_sim.get_states()
                ped_pos = ped_states[-1, human_idx, 0:2]
                dgrp[j] = point_to_segment_dist(ped_pos[0,0], ped_pos[0,1], ped_pos[1,0], ped_pos[1,1], self.robot.px, self.robot.py)
            else:
                human_idx = self.groups[j]
                ped_states, _ = self.psf_sim.get_states()
                ped_pos = ped_states[-1, human_idx, 0:2]
                hull = ConvexHull(ped_pos)
                dists = []
                vert_pos = ped_pos[hull.vertices, :]
                for i in range(len(vert_pos) - 1):
                    dists.append(point_to_segment_dist(vert_pos[i,0], vert_pos[i,1], vert_pos[i+1,0], vert_pos[i+1,1], self.robot.px, self.robot.py))
                dgrp[j] = min(dists)

        # detect group collisions
        coll_grp = np.array([1 if (dgrp[j] < self.collision_dist) else 0 for j in range(self.group_num)])

        # set reward
        reward = self.progress_reward * (self.dgoal[-2] - self.dgoal[-1])
        reward += self.success_reward * reached_goal
        reward -= self.collision_penalty * coll_ped.sum()
        reward -= self.discomfort_penalty * ((self.discomfort_dist - dped) * disc_ped).sum()
        reward -= self.group_penalty * coll_grp.sum()

        # check if simulation done
        if self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collision:
            done = True
            info = Collision()
        elif reached_goal == 1:
            done = True
            info = ReachGoal()
        else:
            done = False
            info = Nothing()

        return ob, reward, done, info

    def render(self, mode='human', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        cmap = plt.cm.get_cmap('hsv', 10)
        robot_color = 'yellow'
        goal_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode == 'human':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                ax.add_artist(human_circle)
            ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            plt.show()
        elif mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius, fill=False, color=cmap(i))
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)
                # add time annotation
                global_time = k * self.time_step
                if global_time % 4 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0] - x_offset, agents[i].center[1] - y_offset,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=14) for i in range(self.human_num + 1)]
                    for time in times:
                        ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.Line2D((self.states[k - 1][0].px, self.states[k][0].px),
                                               (self.states[k - 1][0].py, self.states[k][0].py),
                                               color=robot_color, ls='solid')
                    human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                                                   (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                                                   color=cmap(i), ls='solid')
                                        for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([0], [4], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            ax.add_artist(goal)
            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])

            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)

            # compute attention scores
            if self.attention_weights is not None:
                attention_scores = [
                    plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                             fontsize=16) for i in range(len(self.humans))]

            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(state[0].theta),
                                                             state[0].py + radius * np.sin(state[0].theta))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                for i, human in enumerate(humans):
                    human.center = human_positions[frame_num][i]
                    human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=writer)
            else:
                plt.show()
        else:
            raise NotImplementedError
