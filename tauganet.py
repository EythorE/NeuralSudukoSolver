import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_inputs = 81
n_outputs = 9


## Skilgreyning á tauganeti
reset_graph(seed=None)

X = tf.placeholder(tf.float32, shape=(None, 81), name="Puzzles")
sol = tf.placeholder(tf.int32, shape=(None, 81), name="Solutions")
y = tf.add(sol , tf.constant(-1,sol.dtype), name= "y") # dreg frá einn, labels frá 0 uppí 8

lrelu = tf.nn.leaky_relu
lrelu.alpha=0.1

#hidden1 = tf.layers.dense(X, n_inputs*2, activation=lrelu)
#hidden2 = tf.layers.dense(hidden1, n_inputs, activation=lrelu)
hidden = [
        tf.layers.dense(X, n_outputs*2, activation=lrelu)
        for i in range(81)
]
logits = [
    tf.layers.dense(hidden[i], n_outputs)
    for i in range(81)
]

Y_proba = tf.nn.softmax(logits, axis=2) #(81, batch, 9)
val, ind = tf.nn.top_k(logits, 1)
guess8 = tf.reshape(ind,[-1,81])
guess = tf.add(guess8, tf.constant(1,dtype=guess8.dtype))
##

## Þjálfunar skilgreyning
learning_rate = 0.01

#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:,40], logits=logits[40])
#loss = tf.reduce_mean(xentropy)

lossArr = [
        tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y[:,i], logits=logits[i])#, name='softmax_xentropy_' + i)
        for i in range(81)]

loss = tf.reduce_sum(lossArr)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = [optimizer.minimize(lossArr[i]) for i in range(81)]

cellCorr = [tf.nn.in_top_k(logits[i], y[:,i], 1) for i in range(81)]
correct = tf.reduce_all(cellCorr, axis=1) # er öll þrautin rétt?


#cellCorr = tf.equal(tf.cast(guess8, y.dtype),y)
#correct = tf.reduce_all(cellCorr, axis=1) # er öll þreutin rétt?
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
cellacc = tf.reduce_mean(tf.cast(cellCorr, tf.float32))
##

## Dataset
from sudoku import getSud

suds = 800000   # Fjöldi þrauta í training
sudokus,solutions = getSud(suds)

def getSet(num, maxempty=1): #removes rand[0:maxempty] numbers
    rand = np.random.randint(0, suds, num)
    y = solutions[rand,:]
    #X = sudokus[rand,:]
    X = np.copy(y)
    for i in range(num):
        empty = np.random.randint(0, maxempty) # how many to remove
        X0ind = np.random.randint(0, 81, empty) # indexes set to 0
        X[i,X0ind]=0
    return X,y
##
from sudoku import plotSud
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
batch = 1000
loss_old = np.infty
with tf.Session() as sess:
    init.run()
    saver.restore(sess, ".final/multinums")
    for rm in range(1,3):
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
                    save_path = saver.save(sess, ".final/multinums")
        printValidate(rm)
    

## prófum netið á test gögnum
#from sudoku import plotSud
#import matplotlib.pyplot as plt
with tf.Session() as sess:
    saver.restore(sess, ".final/multinums")
    Xtrain, ytrain = getSet(100,1)
    yprob = Y_proba.eval(feed_dict={X: Xtrain, sol: ytrain})
    #print(asdf)
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    
    #plt.close('all')
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    

file_writer = tf.summary.FileWriter("logs", tf.get_default_graph())
file_writer.close()
