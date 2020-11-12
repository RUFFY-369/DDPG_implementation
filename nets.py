import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class critic(keras.Model):
    def __init__(self, n_actions,fcl1_dims=512,fcl2_dims=512,name="Critic",checkpoint_dir="temp/ddpg"):
        super(critic,self).__init__()
        self.fcl1_dims = fcl1_dims    #dimensions saved as member variables for cleaner code(not really required)
        self.fcl2_dims = fcl2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.model_name+'_Ddpg.h5')

        #BatchNormalization

        self.fcl1= Dense(self.fcl1_dims,activation='relu')

        self.fcl1_norm =tf.keras.layers.BatchNormalization()

        self.fcl2 = Dense(self.fcl2_dims,activation='relu')

        self.fcl2_norm=tf.keras.layers.BatchNormalization()

        self.q =Dense(1,activation=None)

    def call(self,state,action):    #When subclassing the `Model` class, you should implement a `call` method.
        act_val = self.fcl1(tf.concat([state,action],axis=1))      #critic net deals with state-action pair so concatnated
        act_val = self.fcl1_norm(act_val)
        act_val = self.fcl2(act_val)
        act_val =self.fcl2(act_val)
        q= self.q(act_val)

        return q

class actor(keras.Model):
    def __init__(self,fcl1_dims=512, fcl2_dims = 512,n_actions=2,name="Actor",checkpoint_dir="temp/ddpg"):
        super(actor,self).__init__()
        self.fcl1_dims = fcl1_dims    #dimensions saved as member variables for cleaner code(not really required)
        self.fcl2_dims = fcl2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.model_name+'_Ddpg.h5')

        self.fcl1 = Dense(self.fcl1_dims,activation='relu')

        #BatchNormalization
        self.fcl1_norm =tf.keras.layers.BatchNormalization()

        self.fcl2 = Dense(self.fcl2_dims,activation='relu')

        self.fcl2_norm=tf.keras.layers.BatchNormalization()

        self.mu =Dense(self.n_actions,activation='tanh')

    def call(self,state):       #When subclassing the `Model` class, you should implement a `call` method.
        prb= self.fcl1(state)
        prb=self.fcl1_norm(prb)
        prb = self.fcl2(prb)
        prb=self.fcl2_norm(prb)
        mu = self.mu(prb)       #mu(nomenclature as mentioned in the paper)

        return mu
