import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt

import gym


env = gym.make('CartPole-v0')

# Markov Decision  Process
#The agent learns to assign value to actions the lead eventually to the reward

#Policy gradient
# find policy  that maximize Quality(policy) =optimization
#find theta that maximizes J(theta)
# NN is used to learn optimal stohastic
#we can use whatever optimization algo we want(genetic,simulated,monte carlo...)

g = 0.99

def discount_rewards(r):
    #r + g*r+ g^2*r
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * g + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():

    def __init__(self,lr,state_size,action_size,hidden_size):

        self.state=tf.placeholder(shape=[None,state_size],dtype=tf.float32)

        #NN -->> takes a state and produce an action
        hidden_layer = slim.fully_connected(self.state, hidden_size,  activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden_layer, action_size, activation_fn=tf.nn.softmax)
        self.action=tf.argmax(self.output,1)


        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        weights = tf.trainable_variables()
        self.gradient_holders = []

        for idx, var in enumerate(weights):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, weights)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, weights))





m_Agent = agent(lr=1e-2, state_size=4, action_size=2, hidden_size=8)

iter = 5000
max_ep = 1000
update_frequency = 5

total_reward=[]
total_length=[]

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    experiences=sess.run(tf.trainable_variables())

    experiences =np.zeros_like(experiences)


    for i in range(iter):

        state=env.reset()

        score=0
        history=[]


        for j in range(max_ep):

            #choose an action(forward pass of NN)
            act_net=sess.run(m_Agent.output, feed_dict={m_Agent.state: [state]})
            action = np.random.choice(act_net[0], p=act_net[0])
            action = np.argmax(act_net == action)

            #get reward and new state
            state_new,reward,flag,_=env.step(action)

            history.append([state,action,reward,state_new])


            state=state_new
            score+=reward

            #it find target then update gradients experiences and network
            if(flag==True):
                history=np.array(history)
                history[:,2]=discount_rewards(history[:,2])


                #update gradients with all gradient
                #to update the network it uses all gradients(exeriences) of this episode not just the last action
                grads = sess.run(m_Agent.gradients, feed_dict={m_Agent.reward_holder:history[:,2],
                        m_Agent.action_holder:history[:,1],m_Agent.state:np.vstack(history[:,0])})

                for k, grad in enumerate(grads):
                    experiences[k] += grad


                if i%update_frequency ==0 and i!=0:

                    feed_dict = dictionary = dict(zip(m_Agent.gradient_holders, experiences))

                    #train NN
                    sess.run(m_Agent.update_batch, feed_dict=feed_dict)

                    experiences = np.zeros_like(experiences)

                total_reward.append(score)
                total_length.append(j)
                break

        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
