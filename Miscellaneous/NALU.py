import numpy as np
import tensorflow as tf

arithmetic_functions={
'add': lambda x,y :x+y,
'sub': lambda x,y:x-y,
'mul': lambda x,y: x*y,
'div': lambda x,y: x/y,
'square': lambda x,y: np.square(x),
'sqrt' : lambda x,y : np.sqrt(x),
}

def get_data(N, op):
    split = 4
    X_train = np.random.rand(N, 10)*10
    #to be mutually exclusive
    a = X_train[:, :split].sum(1)
    b = X_train[:, split:].sum(1)
    Y_train = op(a, b)[:, None]
    print(X_train.shape)
    print(Y_train.shape)
    
    X_test = np.random.rand(N, 10)*100
    #to be mutually exclusive
    a = X_test[:, :split].sum(1)
    b = X_test[:, split:].sum(1)
    Y_test = op(a, b)[:, None]
    print(X_test.shape)
    print(Y_test.shape)
    
    return (X_train,Y_train),(X_test,Y_test)
  
  

def NALU(prev_layer, num_outputs):
    """ Neural Arithmetic Logic Unit 
    Arguments:
    prev_layer
    num_outputs 
    Returns:
    Output of NALU
    """
    eps=1e-7

    shape = (int(prev_layer.shape[-1]),num_outputs)

    # NAC cell
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    a = tf.matmul(prev_layer, W)
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    
    # NALU
    m = tf.exp(tf.matmul(tf.log(tf.abs(prev_layer) + eps), W))
    g = tf.sigmoid(tf.matmul(prev_layer, G))
    out = g * a + (1 - g) * m

    return out


if  __name__ == "__main__" :


    tf.reset_default_graph()

    train_examples=10000

    (X_train,Y_train),(X_test,Y_test)=get_data(train_examples,arithmetic_functions['add'])  


    X = tf.placeholder(tf.float32, shape=[train_examples, 10])
    Y = tf.placeholder(tf.float32, shape=[train_examples, 1])

    X_1=NALU(X,2)
    Y_pred=NALU(X_1,1)



    loss = tf.nn.l2_loss(Y_pred - Y) # NALU uses mse
    optimizer = tf.train.AdamOptimizer(0.1)
    train_op = optimizer.minimize(loss)



    with tf.Session() as session:
            
        session.run(tf.global_variables_initializer())

        for ep in range(50000):
            

            _,pred,l = session.run([train_op, Y_pred, loss], 
                    feed_dict={X: X_train, Y: Y_train})


            if ep % 1000 == 0:
                print('epoch {0}, loss: {1}'.format(ep,l))

        
        test_predictions,test_loss = session.run([train_op, Y_pred,loss],feed_dict={X:X_test,Y:Y_test})


    print(test_loss)
