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
n_hidden1 = 81*10
n_hidden2 = 42

n_outputs = 81*9

X = tf.placeholder(tf.float32, shape=(None, 81), name="X")
sol = tf.placeholder(tf.int32, shape=(None, 81), name="Solutions")
y = tf.add(sol , tf.constant(-1,sol.dtype), name= "y") # dreg frá einn, labels frá 0 uppí 8

## Nota autocoder lög
reuse=tf.AUTO_REUSE
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          trainable=False, name="hidden_1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                          trainable=False, name="hidden_2")

# næ í breyturnar sem ég vill save-a og bý til saver fyrir þær
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
reuse_saver = tf.train.Saver(reuse_vars_dict)


## Skilgreyning á tauganeti # skoða hvort við þurfum eitthvað regularization
hidden3 = tf.layers.dense(hidden2, n_outputs, activation=tf.nn.relu, name="hidden_3")
#hidden4 = tf.layers.dense(hidden3, n_outputs, activation=tf.nn.relu, name="hidden_4")
#hidden5 = tf.layers.dense(hidden4, n_outputs, activation=tf.nn.relu, name="hidden_5")

    
with tf.name_scope("logits"):
    logits = tf.layers.dense(hidden3, n_outputs, name="logits")
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
        tf.summary.scalar('Loss', loss) #tensorboard
        
    # minnka learning rate þegar netið er að þjálfast 
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.90, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)
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
    #summary fyrir tensorboard
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Cell_Accuracy', cellacc)



## Dataset
suds = 800000   # Fjöldi þrauta í training
if not 'sudokus' in globals(): # nær ekki aftur í þrautir ef til
    sudokus,solutions = getSud(suds)

def getSet(num, maxempty=0): #removes rand[0:maxempty] numbers
    rand = np.random.randint(0, suds, num)
    y = solutions[rand,:]
    #X = sudokus[rand,:] # Þrautir
    X = np.copy(y) # Lausnir sem þrautir
    for i in range(num):
        empty = np.random.randint(0, maxempty+1) # how many to remove
        X0ind = np.random.randint(0, 81, empty) # indexes set to 0
        X[i,X0ind]=0
    return X,y
##


merged = tf.summary.merge_all() #Sameinna öll summaries
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter("./sudoku_fitter/logs/",
                                    tf.get_default_graph())
init = tf.global_variables_initializer()


# Keyrslufasi
print('keyrsla')
epochs=100000
batch = 100
rm = 0          #max empty cells

with tf.Session() as sess:
    init.run()
    
    # Bara nota eitt af þessum
    reuse_saver.restore(sess, "./reuse/save/") #loada sjalfkóðara
    # saver.restore(sess, "./sudoku_fitter/save/") #Loada öllum breytum
    
    Xtrain, ytrain = getSet(batch,rm)
    loss_old = loss.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Starting loss: {:.4f}".format(loss_old))

    for epoch in range(epochs):
        Xtrain, ytrain = getSet(batch,rm)
        sess.run(training_op, feed_dict={X: Xtrain, sol: ytrain})
        if epoch%100 == 0:
            summary, gstep = sess.run([merged, global_step],
                                      feed_dict={X: Xtrain, sol: ytrain})
            file_writer.add_summary(summary, gstep)
            if epoch%1000 == 0:
                Xtrain, ytrain = getSet(batch, rm)
                loss_new, acc_test, cellacc_test, lrate = sess.run(
                        [loss, accuracy, cellacc, learning_rate],
                        feed_dict={X: Xtrain, sol: ytrain})
                if loss_new < loss_old:
                    loss_old = loss_new
                    save_path = saver.save(sess, "./sudoku_fitter/save/")
                    print("Checkpoint Saved:")
                print("{:8d}:  cell acc: {:4.2f}%  accuracy: {:4.2f}%  Loss: {:4.4}  Learning rate: {:.4f}".format(
                        gstep, cellacc_test*100, acc_test*100, loss_new, lrate))
                if epoch%10000 == 0:
                        ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]}) 
                        plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    

    # Validate
    Xtrain, ytrain = getSet(100,rm)
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]}) 
    plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    
    
    

## prófum netið á test gögnum
with tf.Session() as sess:
    saver.restore(sess,"./sudoku_fitter/save/")
    Xtrain, ytrain = getSet(100,rm)
    yprob = Y_proba.eval(feed_dict={X: Xtrain, sol: ytrain})
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    

file_writer.close()
