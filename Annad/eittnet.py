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
n_outputs = 81*9

X = tf.placeholder(tf.float32, shape=(None, 81), name="X")
sol = tf.placeholder(tf.int32, shape=(None, 81), name="Solutions")
y = tf.add(sol , tf.constant(-1,sol.dtype), name= "y") # dreg frá einn, labels frá 0 uppí 8
##

## Skilgreyning á tauganeti
#he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_inputs, activation=tf.nn.relu, name="hidden_1")#, kernel_initializer=he_init, name="hidden_1")
#hidden2 = tf.layers.dense(hidden1, n_inputs, activation=tf.nn.relu, kernel_initializer=he_init, name="hidden_2")
    
with tf.name_scope("logits"):
    logits = tf.layers.dense(hidden1, n_outputs, name="logits")
    logit = tf.reshape(logits,[-1,81,9], name="2D_logits")

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
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
##


with tf.name_scope("Solver"):
    val, ind = tf.nn.top_k(logit)
    guess8 = tf.reshape(ind,[-1,81])
    guess = tf.add(guess8, tf.constant(1,dtype=guess8.dtype))


with tf.name_scope("Validate"):
    cellCorr = [tf.nn.in_top_k(logit[:,i,:], y[:,i], 1) for i in range(81)]
    correct = tf.reduce_all(cellCorr, axis=1) # er öll þrautin rétt?
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    cellacc = tf.reduce_mean(tf.cast(cellCorr, tf.float32))


## Dataset
suds = 800000   # Fjöldi þrauta í training
if not 'sudokus' in globals(): # nær ekki aftur í þrautir ef til
    sudokus,solutions = getSud(suds)

def getSet(num, maxempty=0): #removes rand[0:maxempty] numbers
    rand = np.random.randint(0, suds, num)
    y = solutions[rand,:]
    #X = sudokus[rand,:] # Þrautir
    X = np.copy(y) # Lausnir sem þrautir
    if maxempty > 0:
        for i in range(num):
            empty = np.random.randint(0, maxempty) # how many to remove
            X0ind = np.random.randint(0, 81, empty) # indexes set to 0
            X[i,X0ind]=0
    return X,y
##

def printValidate(rm):
    Xtrain, ytrain = getSet(100,rm)
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
        
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]}) 
    plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# Keyrslufasi
print('keyrsla')
ephocs=2000
batch = 100
loss_old = np.infty
with tf.Session() as sess:
    init.run()
    #saver.restore(sess, "./final/save/")
    file_writer = tf.summary.FileWriter("./eittnet/logs/", tf.get_default_graph())
    for rm in range(0,3):
        print("#####################  {:2d} empty cells  #####################".format(rm))
        Xtrain, ytrain = getSet(batch,rm)
        loss_old = loss.eval(feed_dict={X: Xtrain, sol: ytrain})
        #for epoch in range(ephocs):
        epoch = 0
        acc_test = 0
        count=0
        while(acc_test < 0.99 and count<5):
            if acc_test > 0.99:
                count=count+1
            epoch = epoch + 1
            Xtrain, ytrain = getSet(batch,rm)
            sess.run(training_op, feed_dict={X: Xtrain, sol: ytrain})
            if epoch%100 == 0:
                Xtrain, ytrain = getSet(batch,rm)
                loss_new = loss.eval(feed_dict={X: Xtrain, sol: ytrain})
                if loss_new < loss_old or epoch%10000==0:
                    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
                    cellacc_test = cellacc.eval(feed_dict={X: Xtrain, sol: ytrain})
                    loss_old = loss_new
                    print("rm:{:2d} {:6d}:  cell acc: {:4.2f}%  accuracy: {:4.2f}%  Loss: {:4.4}".format(
                            rm,epoch, cellacc_test*100, acc_test*100, loss_new))
                    save_path = saver.save(sess, "./eittnet/save/")
        printValidate(rm)
    

## prófum netið á test gögnum
with tf.Session() as sess:
    saver.restore(sess, "\eittnet\save")
    Xtrain, ytrain = getSet(100,2)
    yprob = Y_proba.eval(feed_dict={X: Xtrain, sol: ytrain})
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    

file_writer.close()
