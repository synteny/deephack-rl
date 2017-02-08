import os
import threading

import sys
import tensorflow as tf
import numpy as np
import time
from keras import backend as K
import gym
from gym import wrappers

from atari_environment import AtariEnvironment
from config import get_flags
from model import build_policy_and_value_networks

FLAGS = get_flags()

# Shared global parameters
T = 0
TMAX = FLAGS.tmax
t_max = FLAGS.network_update_frequency


def sample_policy_action(probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    # probs = probs - np.finfo(np.float32).epsneg
    #
    # histogram = np.random.multinomial(1, probs)
    # action_index = int(np.nonzero(histogram)[0])
    # return action_index
    return np.random.choice(range(len(probs)), p=probs)


def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Unpack graph ops
    s, a, R, minimize, p_network, v_network = graph_ops

    # Unpack tensorboard summary stuff
    r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

    num_actions = get_num_actions(env)

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height,
                           agent_history_length=FLAGS.agent_history_length)

    time.sleep(5 * num)

    # Set up per-episode counters
    ep_reward = 0
    ep_avg_v = 0
    v_steps = 0
    ep_t = 0

    probs_summary_t = 0


    s_t = env.get_initial_state()
    terminal = False

    while T < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []

        t = 0
        t_start = t

        while not (terminal or ((t - t_start) == t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            probs = session.run(p_network, feed_dict={s: [s_t]})[0]
            action_index = sample_policy_action(probs)
            a_t = np.zeros([num_actions])
            a_t[action_index] = 1

            if probs_summary_t % 100 == 0:
                print("P, ", probs, "V ", session.run(v_network, feed_dict={s: [s_t]})[0][0])

            s_batch.append(s_t)
            a_batch.append(a_t)

            s_t, r_t, terminal, info = env.step(action_index)
            ep_reward += r_t

            r_t = np.clip(r_t, -1., 1.)
            past_rewards.append(r_t)

            t += 1
            T += 1
            ep_t += 1
            probs_summary_t += 1

        if terminal:
            R_t = 0
        else:
            R_t = session.run(v_network, feed_dict={s: [s_t]})[0][0]  # Bootstrap from last state

        R_batch = np.zeros(t)
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + FLAGS.gamma * R_t
            R_batch[i] = R_t

        session.run(minimize, feed_dict={R: R_batch,
                                         a: a_batch,
                                         s: s_batch})

        # Save progress every 5000 iterations
        if T % FLAGS.checkpoint_interval == 0:
            saver.save(session, os.path.join(FLAGS.checkpoint_dir, FLAGS.experiment), global_step=T)

        if terminal:
            # Episode ended, collect stats and reset game
            session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
            print("THREAD:", num, "/ TIME", T, "/ REWARD", ep_reward)
            s_t = env.get_initial_state()
            terminal = False
            # Reset per-episode counters
            ep_reward = 0
            ep_t = 0


def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)
    ep_avg_v = tf.Variable(0.)
    tf.scalar_summary("Episode Value", ep_avg_v)
    val_summary_placeholder = tf.placeholder("float")
    update_ep_val = ep_avg_v.assign(val_summary_placeholder)
    summary_op = tf.merge_all_summaries()
    return r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op


def train(session, graph_ops, saver):
    # Set up game environments (one per thread)
    envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]

    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(os.path.join(FLAGS.summary_dir, FLAGS.experiment), session.graph)

    # Start NUM_CONCURRENT training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver))
                             for thread_id in range(FLAGS.num_concurrent)]
    for t in actor_learner_threads:
        t.start()

    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        if FLAGS.show_training:
            for env in envs:
                env.render()
        now = time.time()
        if now - last_summary_time > FLAGS.summary_interval:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()


def evaluation(session, graph_ops, saver):
    saver.restore(session, FLAGS.checkpoint_path)
    print("Restored model weights from ", FLAGS.checkpoint_path)
    monitor_env = gym.make(FLAGS.game)
    monitor_env = wrappers.Monitor(monitor_env, os.path.join(FLAGS.eval_dir, FLAGS.experiment), force=True)

    # Unpack graph ops
    s, a_t, R_t, minimize, p_network, v_network = graph_ops

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height, agent_history_length=FLAGS.agent_history_length)

    for i_episode in range(100):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            # Forward the deep q network, get Q(s,a) values
            probs = p_network.eval(session = session, feed_dict = {s : [s_t]})[0]
            action_index = sample_policy_action(probs)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)


def build_graph(num_actions):
    # Create shared global policy and value networks
    s, p_logits, v_network, p_params, v_params = build_policy_and_value_networks(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length, resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height)

    # Shared global optimizer
    if FLAGS.use_adam:
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    elif FLAGS.use_rmsprop:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon, decay=FLAGS.alpha, use_locking=False)
    else:
        raise Exception("optimizer not specified")

    # Op for applying remote gradients
    R_t = tf.placeholder("float", [None])
    a_t = tf.placeholder("float", [None, num_actions])
    p_network = tf.nn.softmax(p_logits)

    log_probs = tf.nn.log_softmax(tf.clip_by_value(p_logits, 1e-20, 1.0))
    action_prob = tf.reduce_sum(a_t * log_probs, 1)

    p_loss = -action_prob * (R_t - tf.stop_gradient(v_network))
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    entropy = tf.reduce_sum(p_network * log_probs)
    total_loss = p_loss + 0.5*v_loss + FLAGS.beta*entropy

    minimize = optimizer.minimize(total_loss)
    return s, a_t, R_t, minimize, p_network, v_network

    # s, p_network, v_network, p_params, v_params = \
    # build_policy_and_value_networks(num_actions=num_actions, agent_history_length=FLAGS.agent_history_length,
    #                                 resized_width=FLAGS.resized_width, resized_height=FLAGS.resized_height)
    # # Shared global optimizer
    # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # # optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, decay=FLAGS.alpha, use_locking=False)
    #
    # # Op for applying remote gradients
    # R_t = tf.placeholder("float", [None])
    # a_t = tf.placeholder("float", [None, num_actions])
    #
    # log_pi = tf.log(tf.clip_by_value(p_network, 1e-20, 1.0))
    # entropy = -tf.reduce_sum(p_network * log_pi, reduction_indices=1)
    #
    # # the paper gives formula for gradient ascent, we use a gradient descent optimizer, therefore negate sign:
    # p_loss = -tf.reduce_sum(tf.reduce_sum(tf.mul(log_pi, a_t), reduction_indices=1) * (R_t - v_network) + FLAGS.beta * entropy)
    # v_loss = tf.nn.l2_loss(R_t - v_network)
    #
    # # critic learning rate is half that of the actor
    # total_loss = p_loss + (0.5 * v_loss)
    #
    # minimize = optimizer.minimize(total_loss)
    # return s, a_t, R_t, minimize, p_network, v_network


def get_num_actions(env=None):
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env
    if not env:
        env = gym.make(FLAGS.game)
    num_actions = env.action_space.n
    if (FLAGS.game == "Pong-v0" or FLAGS.game == "Breakout-v0"):
        # Gym currently specifies 6 actions for pong
        # and breakout when only 3 are needed. This
        # is a lame workaround.
        num_actions = 3
    return num_actions


def main(_):
  g = tf.Graph()
  with g.as_default(), tf.Session() as session:
    K.set_session(session)
    num_actions = get_num_actions()
    graph_ops = build_graph(num_actions)
    saver = tf.train.Saver()

    if FLAGS.testing:
        evaluation(session, graph_ops, saver)
    else:
        train(session, graph_ops, saver)

if __name__ == "__main__":
    tf.app.run()
