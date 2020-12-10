import time
import numpy as np

from random_environment import Environment
from agent import Agent


# Main entry point
if __name__ == "__main__":
    file2write=open("tuning4.txt",'a')
    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())

    for i in range(40):
        random_seed = int(time.time())
#        np.random.seed(1606257281)
        np.random.seed(random_seed)
        file2write.write('\n')
        file2write.write('seed: '+ str(random_seed))
        print('\n'+'seed: '+ str(random_seed))
        # Create a random environment
        environment = Environment(magnification=500)
        # Create an agent
        agent = Agent(0.001,0.9,200)
    
        # Get the initial state
        state = environment.init_state
    
        # Determine the time at which training will stop, i.e. in 10 minutes (600 s0e.1.`1`conds) time
        start_time = time.time()
        end_time = start_time + 600
    
        # Train the agent, until the time is up
        while time.time() < end_time and agent.done==False:
            # If the action is to start a new episode, then reset the state
            if agent.has_finished_episode():
                state = environment.init_state
            # Get the state and action from the agent
            action = agent.get_next_action(state)
            # Get the next state and the distance to the goal
            next_state, distance_to_goal = environment.step(state, action)
            # Return this to the agent
            agent.set_next_state_and_distance(next_state, distance_to_goal)
            # Set what the new state is
            state = next_state
            # Optionally, show the environment
            if display_on:
                environment.show(state)
        duration=time.time()-start_time
        # Test the agent for 100 steps, using its greedy policy
        state = environment.init_state
        has_reached_goal = False
        for step_num in range(500):
            action = agent.get_greedy_action(state)
            next_state, distance_to_goal = environment.step(state, action)
            # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
            if distance_to_goal < 0.03:
                has_reached_goal = True
                print(step_num)
                break
            state = next_state
            if display_on:
                environment.show(state)
        file2write.write('\n')
        # Print out the result
        if has_reached_goal:
            file2write.write('Reached goal in ' + str(step_num) + ' steps'+ 'in '+str(duration))
            print('Reached goal in ' + str(step_num) + ' steps'+ 'in '+str(duration))
        else:
            file2write.write('Did not reach goal. Final distance = ' + str(distance_to_goal))
            print('Did not reach goal. Final distance = ' + str(distance_to_goal)+' time: '+str(duration))
                    
                
   
    file2write.close()
