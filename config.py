import tensorflow as tf

def get_flags():
    flags = tf.app.flags

    flags.DEFINE_string('experiment', 'skiing', 'Name of the current experiment')
    flags.DEFINE_string('game', 'Skiing-v0', 'Name of the atari game to play. Full list here: https://gym.openai.com/envs#atari')
    flags.DEFINE_integer('num_concurrent', 16, 'Number of concurrent actor-learner threads to use during training.')
    flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
    flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
    flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
    flags.DEFINE_integer('agent_history_length', 4, 'Use this number of recent screens as the environment state.')
    flags.DEFINE_integer('network_update_frequency', 32, 'Frequency with which each actor learner thread does an async gradient update')
    flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
    flags.DEFINE_float('alpha', 0.99, 'RMSProp decay.')
    flags.DEFINE_float('beta', 1e-2, 'Initial learning rate.')
    flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
    flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal epsilon.')
    flags.DEFINE_string('summary_dir', './summaries', 'Directory for storing tensorboard summaries')
    flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Directory for storing model checkpoints')
    flags.DEFINE_integer('summary_interval', 5,
                         'Save training summary to file every n seconds (rounded '
                         'up to statistics interval.')
    flags.DEFINE_integer('checkpoint_interval', 60,
                         'Checkpoint the model (i.e. save the parameters) every n '
                         'seconds (rounded up to statistics interval.')
    flags.DEFINE_boolean('show_training', False, 'If true, have gym render evironments during training')
    flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')

    flags.DEFINE_string('checkpoint_path', './checkpoint', 'Path to recent checkpoint to use for evaluation')
    flags.DEFINE_string('eval_dir', './eval', 'Directory to store gym evaluation')
    flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
    return flags.FLAGS
