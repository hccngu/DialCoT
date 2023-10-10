import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_LAUNCH_BLOCKING"]  = '1'
os.environ["HF_HOME"] = "./hf_home"
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from OurEnv import OurEnv
import random

# times = 100
# random_list = np.random.randint(0, 500, size=times)


def evaluate_policy(args, env, agent, state_norm):
    # times = 3  # 200
    # random_list = np.random.randint(0, 7470, size=times)
    if args.eval_datasets == 'gsm8k':
        eval_data_path = 'data/gsm8k/gsm8k_test.txt'
    elif args.eval_datasets == 'SVAMP':
        eval_data_path = 'data/zero-shot-cot-test-data/SVAMP/SVAMP_test.txt'
    elif args.eval_datasets == 'MultiArith':
        eval_data_path = 'data/zero-shot-cot-test-data/MultiArith/MultiArith_test.txt'
    else:
        raise NotImplementedError
    with open(eval_data_path, 'r') as f:
            data = f.readlines()

    
    
    evaluate_reward = 0
    count = 0
    right_count = 0
    for line in data:
        # data_index = random.randint(0, 7000)
        line_dic = eval(line)
        previous_string_idx, s, answer = env.only_eval_reset(line_dic)
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            episode_steps += 1
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            print('eval_action:', a, answer[a[0]])
            s_, r, done, previous_string_idx, answer, flag = env.only_eval_step(a, previous_string_idx, episode_steps)
            # print('eval_reward:', r)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        if flag:
            right_count += 1
        evaluate_reward += episode_reward
        count += 1

    return evaluate_reward / count, right_count / count


def main(args, env_name, number, seed):
    # env = gym.make(env_name)
    env = OurEnv(args)
    # env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # env_evaluate = OurEnv(args)
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    # replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    # if args.use_reward_norm:  # Trick 3:reward normalization
    #     reward_norm = Normalization(shape=1)
    # elif args.use_reward_scaling:  # Trick 4:reward scaling
    #     reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # data_index = 0
    # while total_steps < args.max_train_steps:
    #     data_index %= 500
    #     previous_string_idx, s, answer = env.reset(data_index)
    #     if args.use_state_norm:
    #         s = state_norm(s)
    #     if args.use_reward_scaling:
    #         reward_scaling.reset()
    #     episode_steps = 0
    #     done = False
    #     while not done:
    #         episode_steps += 1
    #         a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
    #         print('action:', a, a_logprob, answer[a[0]])
    #         s_, r, done, previous_string_idx, answer = env.step(a, previous_string_idx, episode_steps)
    #         # print('reward:', r)
    #         if args.use_state_norm:
    #             s_ = state_norm(s_)
    #         if args.use_reward_norm:
    #             r = reward_norm(r)
    #         elif args.use_reward_scaling:
    #             r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            # if done and episode_steps != args.max_episode_steps:
            #     dw = True
            # else:
            #     dw = False

            # replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            # s = s_
            # total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            # if replay_buffer.count == args.batch_size:
            #     print('Total steps == {}, updating...'.format(total_steps))
            #     agent.update(replay_buffer, total_steps)
            #     replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            # if total_steps % args.evaluate_freq == 0:
    print('Evaluating...')
    evaluate_num += 1
    evaluate_reward, eval_acc = evaluate_policy(args, env, agent, state_norm)
    # evaluate_rewards.append(evaluate_reward)
    print("evaluate_num:{} \t evaluate_reward:{} \t evaluate_acc:{}".format(evaluate_num, evaluate_reward, eval_acc))
    # writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
    # Save the rewards
    # if evaluate_num % args.save_freq == 0:
    #     np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))
    #     torch.save(agent.actor.state_dict(), 'DRL-code-pytorch/4.PPO-discrete/output_ckpt/PPO_discrete_env_{}_number_{}_seed_{}_actor.ckpt'.format(env_name, number, seed))
    #     torch.save(agent.critic.state_dict(), 'DRL-code-pytorch/4.PPO-discrete/output_ckpt/PPO_discrete_env_{}_number_{}_seed_{}_critic.ckpt'.format(env_name, number, seed))
        # data_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(20480), help=" Maximum number of training steps 2e5")
    parser.add_argument("--evaluate_freq", type=float, default=256, help="Evaluate the policy every 'evaluate_freq' steps 5000")  # 1024
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")  # 5
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size 2048")  # 1024
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size 64")  # 32
    parser.add_argument("--hidden_width", type=int, default=1024, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=8, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")  # True
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")  # True
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")  # True
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    # parser.add_argument('--data', type=str, help='the data used for instructing tuning')
    parser.add_argument('--model_type', default="flan-t5-xl-shared-final1-59-ep", type=str)  # flan-t5-xl-simple-59-ep  shared-final2-59-ep hugging
    parser.add_argument('--model_path', default="google/flan-t5-xl", type=str)
    parser.add_argument('--model_path_decom', default="outputs/flanT5/xl/shared_final1_just_gsm8k/eighty/epoch=59-step=300.ckpt", type=str)
    parser.add_argument('--model_path_solver', default="outputs/flanT5/xl/shared_final1_just_gsm8k/eighty/epoch=59-step=300.ckpt", type=str)
    # parser.add_argument('--lora_weights', default="Alpaca-CoT/saved_models/llama-7b-hf_cot", type=str)
    parser.add_argument('--train_dataset', default="gsm8k", type=str, choices=['gsm8k-200-eval', 'gsm8k', 'SVAMP', 'MultiArith'])
    parser.add_argument('--prompt_type', default="decom2", type=str, choices=['Zero-CoT', 'Few-CoT', 'prompt1', 'prompt2', 'prompt3', 'prompt4', 'decom1', 'decom2', 'decom3'])
    parser.add_argument('--device', default='cuda:5', type=str)
    parser.add_argument('--max_length', default=512, type=int, help='generate max length.')
    parser.add_argument('--max_response_iter', default=5, type=int, help='max response iter for decom3.')
    parser.add_argument('--n_actions', default=3, type=int, help='n actions')
    parser.add_argument('--e_greedy', default=0.9, type=float, help='e greedy')
    parser.add_argument('--np_sample', default=False, type=bool, help='np sample or Categorical().sample()')
    parser.add_argument('--T', default=1.0, type=float, help='Temperature')
    parser.add_argument('--number', default=0, type=int, help='for TensorBoard')
    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--eval_path', default="", type=str)
    parser.add_argument('--eval_datasets', default="SVAMP", type=str, help='gsm8k, SVAMP, MultiArith')
    # parser.add_argument('--eval_data_path', default="data/gsm8k/gsm8k_test.txt", type=str)
    # parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)

    args = parser.parse_args()
    print(args)

    env_name = ['CartPole-v1', 'LunarLander-v2', 'OurEnv-v1', 'OurEnv-v2']
    env_index = 3
    main(args, env_name=env_name[env_index], number=args.number, seed=args.seed)
