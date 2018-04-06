import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_inputs = 81
n_outputs = 9

## Skilgreyning á tauganeti
reset_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
sol = tf.placeholder(tf.int32, shape=(None), name="y")
y = tf.add(sol , tf.constant(-1,sol.dtype)) # dreg frá einn, labels frá 0 uppí 8
lrelu = tf.nn.leaky_relu
lrelu.alpha=0.1

hidden = tf.layers.dense(X, n_outputs*2, activation=lrelu)
logits =tf.layers.dense(hidden, n_outputs)
Y_proba = tf.nn.softmax(logits)
predict = tf.add(tf.argmax(logits,1), tf.constant(1,tf.argmax(logits,1).dtype)) # bæti við einum
##

## Þjálfunar skilgreyning
learning_rate = 0.01
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)


correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
##


## Dataset
field = 40

from sudoku import getSud

suds = 800000   # Fjöldi þrauta í training
sudokus,solutions = getSud(suds)

def getSet(num):
    rand = np.random.randint(0, suds, num)
    y = solutions[rand,:]
    #X = sudokus[rand,:]
    X = np.copy(y)
    #X0ind = np.random.randint(0, 81, 2)
    X0ind = [40]
    X[:,X0ind]=0
    return X,y
##



init = tf.global_variables_initializer()
saver = tf.train.Saver()


print('keyrsla')
# Keyrslufasi
ephocs=2000
batch = 100
loss_old = np.infty
with tf.Session() as sess:
    #saver.restore(sess, "./drasl/multi")
    init.run()
    for epoch in range(ephocs):            
        Xtrain, ytrain = getSet(batch)
        sess.run(training_op, feed_dict={X: Xtrain, sol: ytrain[:,field]})
        if epoch%100 == 0:
            Xtrain, ytrain = getSet(batch)
            loss_new = loss.eval(feed_dict={X: Xtrain, sol: ytrain[:,field]})
            if loss_new < loss_old:
                acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain[:,field]})
                print("{:10d}: accuracy: {:.2f}%  Loss: {:4.4}".format(epoch, acc_test * 100, loss_new))
                save_path = saver.save(sess, "./drasl/multi")
    

## prófum netið á test gögnum
from sudoku import plotSud
import matplotlib.pyplot as plt
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./drasl/multi")
    Xtrain, ytrain = getSet(100)
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain[:,field]})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    
    plt.close('all')
    ypredict = np.copy(Xtrain[:1,:])
    ypredict[0,field] = predict.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    


file_writer = tf.summary.FileWriter("logs", tf.get_default_graph())
file_writer.close()
