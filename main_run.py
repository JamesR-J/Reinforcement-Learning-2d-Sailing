import argparse
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from keras import Sequential
from collections import deque
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import torch

from main_game import Game
from PPO import PPO
# from player_movement import moves





def train(args, env, pic_episode_list):
    random_seed = 0
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + args.name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + args.name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + args.name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + args.name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(args.name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    log_freq = args.max_ep_len * 2
    print_freq = args.max_ep_len * 10
    update_timestep = args.max_ep_len * 4

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max episode number : ", args.max_ep_num)
    print("max timesteps per episode : ", args.max_ep_len)
    print("model saving frequency : " + str(args.save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", env.state_space)
    print("action space dimension : ", env.action_space)
    print("--------------------------------------------------------------------------------------------")
    if args.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", args.action_std)
        print("decay rate of std of action distribution : ", args.action_std_decay_rate)
        print("minimum std of action distribution : ", args.min_action_std)
        print("decay frequency of std of action distribution : " + str(args.action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", args.K_epochs)
    print("PPO epsilon clip : ", args.eps_clip)
    print("discount factor (gamma) : ", args.gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", args.lr_actor)
    print("optimizer learning rate critic : ", args.lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")
    sum_of_rewards = []
    agent = PPO(env, args)
    # next_state = np.reshape(next_state, (1, env.state_space))

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= args.max_ep_num * args.max_ep_len:
        state = env.reset()
        # if e in pic_episode_list:   # was a for loop with e being the i
        #     env.take_pics = True
        # print('Will it save frames = {}'.format(env.take_pics))
        current_ep_reward = 0

        for t in range(1, args.max_ep_len + 1):

            # select action with policy
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action, i_episode, t)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if args.has_continuous_action_space and time_step % args.action_std_decay_freq == 0:
                agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % args.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


    return sum_of_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Test_1')
    parser.add_argument('--epsilon', default=1)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--epsilon_min', default=0.01)
    parser.add_argument('--epsilon_decay', default=0.995)
    parser.add_argument('--lr_actor', default=0.0003)
    parser.add_argument('--lr_critic', default=0.001)
    parser.add_argument('--eps_clip', default=0.2)
    parser.add_argument('--K_epochs', default=80)
    parser.add_argument('--act_function', default='tanh')
    parser.add_argument('--max_ep_num', default=2000)
    parser.add_argument('--max_ep_len', default=10000)
    parser.add_argument('--save_model_freq', default=1e5)
    parser.add_argument('--has_continuous_action_space', default=True)
    parser.add_argument('--action_std', default=0.6)
    parser.add_argument('--action_std_decay_rate', default=0.05)
    parser.add_argument('--min_action_std', default=0.1)
    parser.add_argument('--action_std_decay_freq', default=2.5e5)
    parser.add_argument('--max_steer', default=10)
    parser.add_argument('--team_racing', default=True)
    args = parser.parse_args()
    
    pic_episode_list = [0, 9,   19,   29,   39,   49,   59,   69,   79,   89,   99,
                         109,  119,  129,  139,  149,  159,  169,  179,  189,  199,
                         249,  299,  349,  399,  449,  499,  549,  599,  649,  699,
                         749,  799,  849,  899,  949,  999, 1049, 1099, 1149, 1199,
                        1249, 1299, 1349, 1399, 1449, 1499, 1549, 1599, 1649, 1699,
                        1749, 1799, 1849, 1899, 1949, 1999,
                        ]

    results = dict()
    
    env = Game('ai', args)
    sum_of_rewards = train(args, env, pic_episode_list)
    results[args.name] = sum_of_rewards
    
    #plot_result(results, direct=True, k=20)