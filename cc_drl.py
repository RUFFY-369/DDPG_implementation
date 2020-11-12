import numpy as np 
import tensorflow as tf  
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from replay_buffer import replay_buffer
from nets import actor,critic
from ou_noise import ou_action_noise
#import gym


# Implementation of Deep Deterministic Policy Gradient(DDPG)
# : https://arxiv.org/abs/1509.02971


class agent:
    def __init__(self,alpha=0.001,beta = 0.002,inp_dims = [8],env=None,gamma= 0.99,n_actions=2,
                 max_size=1000000,tau = 0.005,fcl1=512,fcl2=512,batch_size = 64):
                 
        self.gamma =gamma
        self.tau = tau
        self.mem = replay_buffer(max_size,inp_dims,n_actions)
        self.batch_size= batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.actor =actor(n_actions=n_actions,name = "Actor")
        self.critic = critic(n_actions=n_actions,name="Critic")
                 
        self.actor_target = actor(n_actions=n_actions,name="Target_actor")
        self.critic_target = critic(n_actions=n_actions,name="Target_critic")
                 
        self.actor.compile(optimizer=Adam(learning_rate=alpha))     #learning rate is given for formality to compile
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.actor_target.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_target.compile(optimizer=Adam(learning_rate=beta))

        self.noise = ou_action_noise(mu=np.zeros(n_actions))

        self.update_net_params(tau=1)
    
    def update_net_params(self,tau=None):
        if tau is None:
            tau = self.tau

        weights=[]

        targets = self.actor_target.weights
        for i,w in enumerate(self.actor.weights):
            weights.append(w*tau +targets[i]*(1-tau))
        self.actor_target.set_weights(weights)

        weights=[]
        targets=self.critic_target.weights
        for i,w in enumerate(self.critic.weights):
            weights.append(w*tau +targets[i]*(1-tau))
        self.critic_target.set_weights(weights)

    def rem_transition(self,state,action,reward,nw_state,done):
        self.mem.transition_store(state,action,reward,nw_state,done)

    def model_save(self):
        print("..........Saving the model..........")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.actor_target.save_weights(self.actor_target.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.critic_target.save_weights(self.critic_target.checkpoint_file)

    def load_model(self):
        print("...........Loading the model..........")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.actor_target.load_weights(self.actor_target.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.critic_target.load_weights(self.critic_target.checkpoint_file)

    def action_choose(self,obsv,evaluate=False):
        states = tf.convert_to_tensor([obsv],dtype=tf.float32)
        actions = self.actor(states)
        #if not evaluate:                                               #they used Ornstein Uhlenbeck(can add class for this noise)
            
            #actions+= tf.random.normal(shape=[self.n_actions],mean=0.0,stddev=0.1)
        #actions = tf.clip_by_value(actions,self.min_action,self.max_action)

        noise =self.noise()                #
        actions_prime = actions + noise    #  #used Ornstein Uhlenbeck
        return actions_prime[0]            #
        #return actions[0]

    def learning(self):
        if self.mem.count_mem < self.batch_size:
            return
        
        state,action,reward,nw_state,done = self.mem.sample_buffer(self.batch_size)
        states= tf.convert_to_tensor(state,dtype=tf.float32)
        nw_states = tf.convert_to_tensor(nw_state,dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward,dtype=tf.float32)
        actions = tf.convert_to_tensor(action,dtype= tf.float32)

        with tf.GradientTape() as tape:
            actions_target = self.actor_target(nw_states)
            nw_critic_val = tf.squeeze(self.critic_target(nw_states,actions_target),1)
            critic_val=tf.squeeze(self.critic(states,actions),1)
            target = reward + self.gamma*nw_critic_val*(1-done)
            critic_loss = keras.losses.MSE(target,critic_val)

        critic_net_grad = tape.gradient(critic_loss,self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(critic_net_grad,self.critic.trainable_variables))


        with tf.GradientTape() as tape:
            new_pol_acs = self.actor(states)
            actor_loss = -self.critic(states,new_pol_acs)
            actor_loss = tf.math.reduce_mean(actor_loss)
        
        actor_net_grad = tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_net_grad,self.actor.trainable_variables))

        self.update_net_params()



    
