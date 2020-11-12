import gym 
from gym import wrappers
import numpy as np
from cc_drl import agent
from utils import plot_learning_curve


# Implementation of Deep Deterministic Policy Gradient(DDPG)
# : https://arxiv.org/abs/1509.02971


if __name__ == "__main__":
    
    
    #Uncomment the env_name for experiments on different environments
    #env_name="InvertedPendulum-v1"
    #env_name = "HalfCheetah-v1"
    #env_name="LunarLanderContinuous-v2"
    #env_name="Humanoid-v1"
    env_name= "Pendulum-v0"
    env =gym.make(env_name)
    agent1 =agent(inp_dims=env.observation_space.shape,env = env,n_actions=env.action_space.shape[0])

    #keeping episode_id :True will record for all the episodes which would consume a lot of memory in case of long training
    #so recording every 25th episode to keep track of the training progress
    env = wrappers.Monitor(env, 'temp/video', video_callable=lambda episode_id: episode_id%25==0 , force=True)

    n_game = 250


    filename = env_name+".png"
    fig_file = "plots/"+ filename
    best_score = env.reward_range[0]
    score_hist= []
    load_checkpoint = False
    env.render( "rgb_array")
    

    if load_checkpoint:
        n_steps = 0
        while n_steps<= agent1.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            nw_observation, reward, done, _ = env.step(action)
            agent1.rem_transition(observation, action, reward, nw_observation, done)
            n_steps += 1
        agent1.learning()
        agent1.load_model()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_game):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent1.action_choose(observation, evaluate)
            nw_observation, reward, done, _ = env.step(action)
            score += reward
            agent1.rem_transition(observation, action, reward, nw_observation, done)
            if not load_checkpoint:
                agent1.learning()
            observation = nw_observation

        score_hist.append(score)
        avg_score = np.mean(score_hist[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent1.model_save()

        print('episode :', i, 'score %.1f :' % score, 'avg score %.1f :' % avg_score)


    


    if not load_checkpoint:
        x = [i+1 for i in range(n_game)]
        plot_learning_curve(x, score_hist, fig_file,n_game)