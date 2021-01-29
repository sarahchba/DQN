import numpy as np
import torch
import collections

class Network(torch.nn.Module):
    

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=200)
        self.layer_3 = torch.nn.Linear(in_features=200, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        #target network
        self.target_network=Network(input_dimension=2, output_dimension=4)
        
    def get_forward_values(self,state):
        state_tensor=torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        return self.q_network.forward(state_tensor)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, batch,update=False):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(batch,update)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, batch, update=True):
        state_array=np.zeros([len(batch),2])
        reward_array=np.zeros(len(batch))
        action_array=np.zeros([len(batch),2])
        next_state_array=np.zeros([len(batch),2])
        for i in range(len(batch)):
            state_array[i]=batch[i][0]
            reward_array[i]=batch[i][2]
            action_array[i]=batch[i][1]
            next_state_array[i]=batch[i][3]
        
        #get discrete action array
        discrete_action_array=np.zeros([len(batch),1])
        for i in range(len(action_array)):
            discrete_action_array[i]=self.get_discrete_action(action_array[i])
            
        minibatch_state_tensor=torch.tensor(state_array,dtype=torch.float32)
        minibatch_next_state_tensor=torch.tensor(next_state_array,dtype=torch.float32)
        reward_tensor=torch.tensor(reward_array,dtype=torch.float32)
        mini_batch_action_tensor=torch.tensor(discrete_action_array,dtype=torch.long)
        

        state_action_q_values, indices=torch.max(self.target_network.forward(minibatch_next_state_tensor),dim=1)  
        state_action_q_values=state_action_q_values.detach()
        
        return_tensor=reward_tensor+state_action_q_values*0.9
        #print(reward_tensor)
        predicted_q_value_tensor=self.q_network.forward(minibatch_state_tensor).gather(dim=1,index=mini_batch_action_tensor).squeeze(-1)
        loss = torch.nn.MSELoss()(predicted_q_value_tensor, return_tensor)
        
        #update target network
        if update:
            self.target_network.load_state_dict(self.q_network.state_dict())
 
        return loss
    
    #get continous action from discrete intervals of actions
    def get_discrete_action(self,continuous_action):
        action=0
        if continuous_action[1]>0.015:
            action=0
        elif continuous_action[0]>0.015:
            action=1
        elif continuous_action[1]<-0.015:
            action=2
        elif continuous_action[0]<-0.015:
            action=3

                
        return action
    
class ReplayBuffer:
    def __init__(self):
        self.buffer=collections.deque(maxlen=1000000)
        self.buffer2=collections.deque(maxlen=10000)
        self.count=0
        self.multiplesOf85=[]
        for i in range(0,6000):
            if i%85==0:
                self.multiplesOf85.append(i)
        
    def append_transition(self,transition):
        self.buffer.append(transition)

        
    def sample_minibatch(self):
        random_batch=collections.deque(maxlen=80)
        if (len(self.buffer)>=6000):
            self.buffer2=[self.buffer[i] for i in range(self.count,len(self.buffer))]
            self.count+=1
            for i in self.multiplesOf85:
                random_batch.append(self.buffer2[i])
        return random_batch
    


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1500
        self.episode=0
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        
        self.epsilon=0.9
        self.epsilon1=0.9
        self.epsilon2=0.9

        #DQN
        self.dqn=DQN()
        self.replay_buffer=ReplayBuffer()
        self.losses=[]
        self.loss_per_episode=[]        

        #Flags for testing with the greedy policy
        self.done=False
        self.test=False
        
        #Flag for updating the target network
        self.target_step=200
        
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        
        #Test with the greedy policy every 5 episodes
        if (self.episode%5==0 and self.episode>10) or self.done==True:
            action=self.get_greedy_action(state)
            self.test=True
        else:    
            #updating epsilon
            if ( self.num_steps_taken % self.episode_length==0) :
                self.epsilon1*=0.99
                self.epsilon=self.epsilon1
            elif self.num_steps_taken % self.episode_length==500:
                self.epsilon2*=0.999
                self.epsilon=self.epsilon2

            #setting the action from the epsilon greedy policy
            q_values=self.dqn.get_forward_values(state).detach().numpy()
            action_probabilities=[]
            for a in range(4):
                if a==np.argmax(q_values):
                    action_probabilities.append(1-self.epsilon+self.epsilon/4)
                else:
                    action_probabilities.append(self.epsilon/4)
            discrete_action=np.random.choice(np.arange(0,4),p=action_probabilities)
            action=self.get_continous_action(discrete_action)
                

        # Update the number of steps, episode and length of episode which the agent has taken
        self.num_steps_taken += 1
        if self.num_steps_taken%self.episode_length==0 :
            if self.episode%5==0 and self.episode_length>300:
                self.episode_length-=200
            self.episode+=1

        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward=10*(1 - distance_to_goal)**3

        #stop training if the agent reaches the goal during greedy policy test
        if self.test==True or self.done==True:
            if distance_to_goal<=0.08:
                self.done=True
            self.test=False
            
        else:   
            # Create a transition
            transition = (self.state, self.action, reward, next_state)
            #append the transition to the replay buffer and sample a minibatch
            self.replay_buffer.append_transition(transition)
            minibatch=self.replay_buffer.sample_minibatch()
            
           #update target network 
            if self.num_steps_taken%self.target_step==0:
                update=True
            else:
                update=False
            
            #train the q-network with the minibatch
            if len(minibatch)!=0:
                self.dqn.train_q_network(minibatch,update)#update
                
                
    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        q_values=self.dqn.get_forward_values(state).detach().numpy()
        action = np.array([0.02, 0.0], dtype=np.float32)
        discrete_action=np.argmax(q_values)
        action=self.get_continous_action(discrete_action)
        return action

    #get continous action from discrete intervals of actions
    def get_continous_action(self,discrete_action):
        action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        if discrete_action==0:
            action[1]=0.02
            action[0]=0.0
        elif discrete_action==1:
            action[0]=0.02
            action[1]=0.0
        elif discrete_action==2:
            action[0]=0.0
            action[1]=-0.02
        elif discrete_action==3:
            action[1]=0.0
            action[0]=-0.02

        return action


    
