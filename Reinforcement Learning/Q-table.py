import numpy as np
import tensorflow as tf
import gym


lr = 0.8
g = 0.9
episodes = 2000

env = gym.make('FrozenLake-v0')

#initalize Q tabe with zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

rList = []

for i in range(episodes):

    s = env.reset()
    reward = 0
    goal_flag = False


    for j in range(200):

        # greedy action
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

        s_new,r,goal_flag,_ = env.step(a)#state and reward


        maxQ=np.max(Q[s_new,:])
        # Belmann
        Q[s,a] += lr*(r + g*maxQ - Q[s,a])

        reward += r
        s = s_new

        if goal_flag == True:
            break

    rList.append(reward)

print ("Score:" +  str(sum(rList)/episodes))

print (" Q-Table ")
print (Q)