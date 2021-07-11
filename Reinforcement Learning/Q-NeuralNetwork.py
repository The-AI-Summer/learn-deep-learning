import numpy as np
import tensorflow as tf
import gym




##Approximate Q values by NN

env = gym.make('FrozenLake-v0')

y = .99
e = 0.1
episodes = 2000


inputs= tf.placeholder(shape=[1,16],dtype=tf.float32)
weights= tf.Variable(tf.random_uniform([16,4],0,0.01))


rList = []

#NN
Q = tf.matmul(inputs,weights)
pred = tf.argmax(Q,1)
Q_target = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss_fun = tf.reduce_sum(tf.square(Q_target - Q))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = optimizer.minimize(loss_fun)



init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for i in range(episodes):

        s = env.reset()
        reward = 0
        goal_flag = False

        for j in range(200):

            #Choose action
            a,allQ = sess.run([pred,Q],feed_dict={inputs:np.identity(16)[s:s+1]})

            #e-greedy
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()#random action


            s_new,r,goal_flag,_ = env.step(a[0])# new state

            # Q values from NN(only forward)
            Q1 = sess.run(Q,feed_dict={inputs:np.identity(16)[s_new:s_new+1]})

            #get maxQ and set target
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1

            #Train NN to learn Q values
            sess.run([updateModel,weights],feed_dict={inputs:np.identity(16)[s:s+1],Q_target:targetQ}) #

            reward += r
            s = s_new

            if goal_flag == True:

                e = 1.0/((i/50) + 10) #Decrese prob
                break

        rList.append(reward)
print ("Score " + str(sum(rList)/episodes) + "%")