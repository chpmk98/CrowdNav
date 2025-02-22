import logging
import copy
import torch
import numpy as np
import pandas as pd
from crowd_nav.policy.gap import GAP
from crowd_sim.envs.utils.info import *


class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        # build an action space if we need it
        if isinstance(target_policy, GAP):
            target_policy.build_action_space(robot.v_pref)
            self.GAP_action_space = target_policy.action_space
        elif isinstance(robot.policy, GAP):
            robot.policy.build_action_space(robot.v_pref)
            self.GAP_action_space = robot.policy.action_space
        self.doPPO = False

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        average_speeds = []
        max_speeds = []
        average_accels = []
        max_accels = []
        average_jerks = []
        max_jerks = []
        # sometimes we do wonky PPO stuff
        self.doPPO = isinstance(self.target_policy, GAP) or isinstance(self.robot.policy, GAP)
        nominal_log_pi = np.log(0.7) # log probability of action used for imitation learning

        with torch.no_grad():
            for i in range(k):
                ob = self.env.reset(phase)
                done = False
                states = []
                roboStates = []
                actions = []
                rewards = []
                vals = []
                log_pis = []
                aas = []
                while not done:
                    if self.doPPO:
                        # generate nominal values if doing imitation learning
                        try:
                            a, action, _, val, log_pi = self.robot.act(ob)
                            vals.append(val)
                        except ValueError:
                            action = self.robot.act(ob)
                            act_norms = np.array([np.sqrt((ass.vx - action.vx)**2 + (ass.vy - action.vy)**2) for ass in self.GAP_action_space])
                            a = np.argmin(act_norms)
                            action = self.GAP_action_space[a]
                            log_pi = nominal_log_pi
                        aas.append(a)
                        log_pis.append(log_pi)
                    else:
                        action = self.robot.act(ob)
                    ob, reward, done, info = self.env.step(action)
                    states.append(self.robot.policy.last_state)
                    actions.append(action)
                    rewards.append(reward)
                    roboStates.append([self.robot.vx, self.robot.vy])
                    # roboStates.append([float(thingy) for thingy in str(self.robot.policy.last_state.self_state).split(' ')])

                    if isinstance(info, Danger):
                        too_close += 1
                        min_dist.append(info.min_dist)

                if isinstance(info, ReachGoal):
                    success += 1
                    success_times.append(self.env.global_time)
                elif isinstance(info, Collision):
                    collision += 1
                    collision_cases.append(i)
                    collision_times.append(self.env.global_time)
                elif isinstance(info, Timeout):
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(self.env.time_limit)
                else:
                    raise ValueError('Invalid end signal from environment')

                roboStates = np.array(roboStates)
                roboDF = pd.DataFrame(roboStates, columns=['vx', 'vy'])
                timeStep = self.env.time_step
                roboDF[['ax', 'ay']] = roboDF[['vx', 'vy']].diff()/timeStep
                roboDF[['jx', 'jy']] = roboDF[['ax', 'ay']].diff()/timeStep
                roboDF['speed'] = np.sqrt(np.square(roboDF[['vx', 'vy']]).sum(axis=1))
                roboDF['accel'] = np.sqrt(np.square(roboDF[['ax', 'ay']]).sum(axis=1))
                roboDF['jerk'] = np.sqrt(np.square(roboDF[['jx', 'jy']]).sum(axis=1))
                average_speeds.append(np.mean(roboDF['speed']))
                max_speeds.append(np.max(roboDF['speed']))
                average_accels.append(np.mean(roboDF['accel']))
                max_accels.append(np.max(roboDF['accel']))
                average_jerks.append(np.mean(roboDF['jerk']))
                max_jerks.append(np.max(roboDF['jerk']))

                # calculate advantages for ppo using GAE (Generalized Advantage Estimation)
                if self.doPPO:
                    # set some GAE parameters
                    gamma = 0.99
                    lam = 0.95
                    try:
                        _, _, _, last_val, _ = self.robot.act(ob)
                    except ValueError:
                        vals = [self._compute_value(ind, rewards) for ind in range(len(states))]
                        action = self.robot.act(ob)
                        _, last_val, _, _ = self.env.step(action)
                        
                    last_advantage = 0
                    advantages = []
                    for t in reversed(range(len(rewards))):
                        delta = rewards[t] + gamma * last_val - vals[t]
                        last_advantage = delta + gamma * lam * last_advantage
                        advantages.append(last_advantage)
                        last_val = vals[t]
                    # reverse the list of advantages because we built it backwards
                    advantages.reverse()

                if update_memory:
                    if isinstance(info, ReachGoal) or isinstance(info, Collision):
                        # only add positive(success) or negative(collision) experience in experience set
                        if self.doPPO:
                            self.update_memory(states, actions, rewards, imitation_learning, aas, vals, log_pis, advantages)
                        else:
                            self.update_memory(states, actions, rewards, imitation_learning)

                cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                               * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        #'''
        ave_ave_speed = sum(average_speeds) / len(average_speeds)
        ave_max_speed = sum(max_speeds) / len(max_speeds)
        ave_ave_accel = sum(average_accels) / len(average_accels)
        ave_max_accel = sum(max_accels) / len(max_accels)
        ave_ave_jerk = sum(average_jerks) / len(average_jerks)
        ave_max_jerk = sum(max_jerks) / len(max_jerks)
        #'''

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))

        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))
            #'''
            logging.info('ave speed: {:.2f}, max speed: {:.2f}, ave accel: {:.2f}, max accel: {:.2f}, ave jerk: {:.2f}, max jerk: {:.2f}'.
                          format(ave_ave_speed, ave_max_speed, ave_ave_accel, ave_max_accel, ave_ave_jerk, ave_max_jerk))
            #'''

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def _compute_value(self, i, rewards):
        value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * rewards[i]
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
        return value

    def update_memory(self, states, actions, rewards, imitation_learning=False, aas=None, vals=None, log_pis=None, advantages=None):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = self._compute_value(i, rewards)
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            if self.doPPO:
                '''
                if imitation_learning:
                    self.memory.push((state, actions[i], value, log_pis[i], advantages[i]))
                else:
                '''
                self.memory.push((state, aas[i], actions[i], vals[i], log_pis[i], advantages[i]))
            else:
                self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
