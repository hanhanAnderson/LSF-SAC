from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args


if __name__ == '__main__':
    for i in range(1):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)

        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
        # window_size_x=640,
        # window_size_y=480,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        print(args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
"""ipconfig /release & ping -n 2 127.0.0.1 & ipconfig /renew
Namespace(alg='qmix', alpha=1.0, anneal_epsilon=1.8999999999999998e-05, batch_size=32, 
buffer_size=5000, comm_beta=0.001, comm_beta_end_decay=50000000, comm_beta_start_decay=20000000, 
comm_beta_target=0.01, comm_embed_dim=3, comm_entropy_beta=1e-06, comm_entropy_beta_end_decay=50000000, 
comm_entropy_beta_start_decay=20000000, comm_entropy_beta_target=0.0001, cuda=True, difficulty='7', 
entropy_coefficient=0.001, episode_limit=150, epsilon=1, epsilon_anneal_scale='step', evaluate=False, 
evaluate_cycle=5000, evaluate_epoch=32, game_version='latest', gamma=0.99, grad_norm_clip=10, hyper_hidden_dim=64, 
is_comm_beta_decay=False, is_comm_entropy_beta_decay=False, lambda_mi=0.001, lambda_nopt=1, lambda_opt=1, lambda_ql=1, 
last_action=True, load_model=False, lr=0.0005, map='3s5z', min_epsilon=0.05, model_dir='./model', n_actions=14, 
n_agents=8, n_episodes=1, n_steps=2000000, noise_dim=16, obs_shape=128, optimizer='RMS', qmix_hidden_dim=32, 
qtran_hidden_dim=64, replay_dir='./replay', result_dir='./result', reuse_network=True, rnn_hidden_dim=64, 
save_cycle=5000, seed=123, state_shape=216, step_mul=8, target_update_cycle=200, train_steps=1, two_hyper_layers=False)

"""