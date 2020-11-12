import numpy as np  
class replay_buffer:
    def __init__(self,max_size,inp_shape,n_actions):
        self.size_mem=max_size          #size of memory buffer
        self.count_mem=0
        self.state_mem=np.zeros((self.size_mem, *inp_shape))  #unpacking
        self.new_state_mem = np.zeros((self.size_mem, *inp_shape))
        self.action_mem = np.zeros((self.size_mem, n_actions))
        self.terminal_mem = np.zeros(self.size_mem,dtype=np.bool)  #to track the done flags 
        self.reward_mem = np.zeros(self.size_mem)

    def transition_store(self,state,action,reward,nw_state,done):
        idx = self.count_mem % self.size_mem
        self.state_mem[idx] = state
        self.new_state_mem[idx] = nw_state
        self.reward_mem[idx] = reward
        self.action_mem[idx] = action
        self.terminal_mem[idx] = done

        self.count_mem+=1

    def sample_buffer(self,batch_size):
        max_mem= min(self.count_mem,self.size_mem)
        batch = np.random.choice(max_mem,batch_size)   #from 0 to max_mem of batch_size

        states = self.state_mem[batch]
        nw_states = self.new_state_mem[batch]
        rewards = self.reward_mem[batch]
        dones = self.terminal_mem[batch]
        actions = self.action_mem[batch]

        return states,actions,rewards,nw_states,dones


