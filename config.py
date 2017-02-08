import multiprocessing
import argparse

def get_flags():
    parser = argparse.ArgumentParser(description='Atari RL training and evaluation')

    parser.add_argument('--experiment', type=str, default='skiing', help='Name of the current experiment')
    parser.add_argument('--game', required=True, type=str, default='Skiing-v0', help='Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
    parser.add_argument('--num_concurrent', type=int, default=multiprocessing.cpu_count(), help='Number of concurrent actor-learner threads to use during training.')
    parser.add_argument('--tmax', type=int, default=80000000, help='Number of training timesteps.')
    parser.add_argument('--resized_width', type=int, default=84, help='Scale screen to this width.')
    parser.add_argument('--resized_height', type=int, default=84, help='Scale screen to this height.')
    parser.add_argument('--agent_history_length', type=int, default=4, help='Use this number of recent screens as the environment state.')
    parser.add_argument('--network_update_frequency', type=int, default=32, help='Frequency with which each actor learner thread does an async gradient update')
    parser.add_argument('--learning_rate', type=float, default=7e-4, help='Initial learning rate.')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSProp decay.')
    parser.add_argument('--beta', type=float, default=1e-2, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount rate.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='RMSProp epsilon.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--use_rmsprop', action='store_true', help="Use RMSProp")
    group.add_argument('--use_adam', action='store_true', help="Use Adam optimizer")
    parser.add_argument('--anneal_epsilon_timesteps', type=int, default=1000000, help='Number of timesteps to anneal epsilon.')
    parser.add_argument('--summary_dir', type=str, default='./summaries', help='Directory for storing tensorboard summaries')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for storing model checkpoints')
    parser.add_argument('--summary_interval', type=int, default=5, help='Save training summary to file every n seconds (rounded up to statistics interval.')
    parser.add_argument('--checkpoint_interval', type=int, default=60, help='Checkpoint the model (i.e. save the parameters) every n '
                         'seconds (rounded up to statistics interval.')
    parser.add_argument('--show_training', action='store_true', help='If true, have gym render evironments during training')
    parser.add_argument('--testing', action='store_true', help='If true, run gym evaluation')

    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='Path to recent checkpoint to use for evaluation')
    parser.add_argument('--eval_dir', type=str, default='./eval', help='Directory to store gym evaluation')
    parser.add_argument('--num_eval_episodes', type=int, default=100, help='Number of episodes to run gym evaluation.')
    return parser.parse_args()
