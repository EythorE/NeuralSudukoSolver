import numpy as np
import tensorflow as tf
from sudoku import getSud
from sudoku import plotSud


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph(seed=None)

## Innmerki og þjálfunargögn
n_inputs = 81
n_outputs = 81*10

X = tf.placeholder(tf.float32, shape=(None, 81), name="X")
y = tf.placeholder(tf.int32, shape=(None, 81), name="y")
#y = tf.add(sol , tf.constant(-1,sol.dtype), name= "y") # dreg frá einn, labels frá 0 uppí 8
##

## Skilgreyning á tauganeti
hidden1 = tf.layers.dense(X, n_outputs, activation=tf.nn.relu, name="hidden_1")
hidden2 = tf.layers.dense(hidden1, 42, activation=tf.nn.relu,name="hidden_2")#n_inputs virkaði
hidden3 = tf.layers.dense(hidden1, n_outputs, activation=tf.nn.relu, name="hidden_3")

#tók út
#he_init = tf.contrib.layers.variance_scaling_initializer()
#kernel_initializer=he_init
    
with tf.name_scope("logits"):
    logits = tf.layers.dense(hidden1, n_outputs, name="logits")
    logit = tf.reshape(logits,[-1,81,10], name="2D_logits")

Y_proba = tf.nn.softmax(logit, name="softmax")
##

## Þjálfunar skilgreyning
with tf.name_scope("Training"):
    #reikna loss sem summu cross entropy af reit
    with tf.name_scope("loss"):
        lossArr = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y[:,i], logits=logit[:,i,:])
                for i in range(81)]
        loss = tf.reduce_sum(lossArr)
    learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
##


with tf.name_scope("Solver"):
    val, ind = tf.nn.top_k(logit)
    guess = tf.reshape(ind,[-1,81])
    #guess = tf.add(guess8, tf.constant(1,dtype=guess8.dtype))


with tf.name_scope("Validate"):
    cellCorr = [tf.nn.in_top_k(logit[:,i,:], y[:,i], 1) for i in range(81)]
    correct = tf.reduce_all(cellCorr, axis=1) # er öll þrautin rétt?
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    cellacc = tf.reduce_mean(tf.cast(cellCorr, tf.float32))


## Dataset
suds = 800000   # Fjöldi þrauta í training
if not 'sudokus' in globals(): # nær ekki aftur í þrautir ef til
    sudokus,solutions = getSud(suds)

def getSet(num): #removes rand[0:maxempty] numbers
    rand = np.random.randint(0, suds, num)
    #y = solutions[rand,:]
    X = sudokus[rand,:] # Þrautir
    y = np.copy(X) # Lausnir sem þrautir
    return X,y
##

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# Keyrslufasi
print('keyrsla')
epochs=20000
batch = 1000
loss_old = np.infty
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./breittnet/save/")
    file_writer = tf.summary.FileWriter("./breittnet/logs/", tf.get_default_graph())
    for epoch in range(epochs):
        Xtrain, ytrain = getSet(batch)
        sess.run(training_op, feed_dict={X: Xtrain, y: ytrain})
        if epoch%100 == 0:
            Xtrain, ytrain = getSet(batch)
            loss_new = loss.eval(feed_dict={X: Xtrain, y: ytrain})
            acc_test = accuracy.eval(feed_dict={X: Xtrain, y: ytrain})
            cellacc_test = cellacc.eval(feed_dict={X: Xtrain, y: ytrain})
            if loss_new < loss_old:
                loss_old = loss_new
                save_path = saver.save(sess, "./breittnet/save/")
                ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]}) 
                plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
                print("Checkpoint Saved:")
            print("{:6d}:  cell acc: {:4.2f}%  accuracy: {:4.2f}%  Loss: {:4.4}".format(
                    epoch, cellacc_test*100, acc_test*100, loss_new))

    

## prófum netið á test gögnum
#from sudoku import plotSud
#import matplotlib.pyplot as plt
with tf.Session() as sess:
    saver.restore(sess,"./breittnet/save/")
    Xtrain, ytrain = getSet(100)
    yprob = Y_proba.eval(feed_dict={X: Xtrain, y: ytrain})
    acc_test = accuracy.eval(feed_dict={X: Xtrain, y: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    
    #plt.close('all')
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    

#file_writer = tf.summary.FileWriter("./final/logs", tf.get_default_graph())
file_writer.close()
