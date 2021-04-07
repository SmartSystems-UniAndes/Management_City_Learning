import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense, BatchNormalization, Concatenate


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic',
                 chkpt_dir='tmp/ddpg', n_actions=1, n_states=3, optimizer="Adam"):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_ddpg.h5")
        #_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.batch1 = BatchNormalization()
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.batch2 = BatchNormalization()
        
        self.fc1a= Dense(250, activation='relu')
        self.batch1a= BatchNormalization()
        
        self.fcout1 = Dense(self.fc2_dims, activation = 'relu')
        
        
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        state_nn = self.fc1((tf.concat([state, action], axis=1)))
        state_nn = self.batch1(state_nn)
        state_nn = self.fc2(state_nn)
        state_nn = self.batch2(state_nn)
        
        #action_nn = self.fc1a(action)
        #action_nn = self.batch1a(action_nn)
        
        
        #concat = Concatenate()([state_nn, action_nn])
        
        #q = self.fcout1(concat)
        q = self.q(state_nn)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + "_ddpg.h6")
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh', kernel_initializer=w_init)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        # if action bounds not +/-1, can multiply here
        mu = self.mu(prob)*0.33333334
        return mu