from DDQN_Networks import CriticNetwork, ActorNetwork
from OUActionNoise import OUActionNoise
from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np

class Agent:
    def __init__(self, input_dims, actor_lr = 0.001, critic_lr = 0.002, env = None, gamma = 0.99, 
                 n_actions = 2, max_size = 1000000, tau = 0.005, fc1 = 400, fc2 = 300, 
                 batch_size = 64, noise = 0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = OUActionNoise(mu = np.zeros(n_actions))
        #self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        self.actor = ActorNetwork(n_actions = n_actions, name = 'actor')
        self.critic = CriticNetwork(name = 'critic')
        self.target_actor = ActorNetwork(n_actions = n_actions, name = 'target_actor')
        self.target_critic = CriticNetwork(name = 'target_critic')
        
        self.actor.compile(optimizer=Adam(learning_rate = actor_lr))
        self.critic.compile(optimizer = Adam(learning_rate = critic_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate = actor_lr))
        self.target_critic.compile(optimizer = Adam(learning_rate = critic_lr))
        
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):  # this is done to update the targets based on a value tau
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau)) # first iteration is a hard copy
        self.target_actor.set_weights(weights)
        
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau)) # first iteration is a hard copy
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('.... load models ....')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate = False):# here we add the noise for the exploration
        state = tf.convert_to_tensor([observation], dtype = tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += self.noise()#tf.random.normal(shape=[self.n_actions],
                        #               mean = 0.0, stddev = self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape: #load up operations up to computational graph
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma*critic_value_*(1-done)
            #critic_loss = keras.losses.MSE(target, critic_value)
            critic_loss = tf.math.reduce_mean(tf.math.square(target - critic_value))
            
        # we calculate the gradients with tape and update the model weights
        critic_network_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        
        actor_network_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradients, self.actor.trainable_variables))
        
        self.update_network_parameters()
