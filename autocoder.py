## Sjálfkóðari

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
n_hidden2 = 70
n_outputs = 81*10

X = tf.placeholder(tf.float32, shape=(None, 81), name="X")
labels=tf.cast(X, dtype=tf.int32)

## Kóðari layerar sem við endurnýtum
hidden1 = tf.layers.dense(X, n_outputs, activation=tf.nn.relu, name="hidden_1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,name="hidden_2")#n_inputs virkaði

# næ í breyturnar sem ég vill save-a og bý til saver fyrir þær
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
reuse_saver = tf.train.Saver(reuse_vars_dict)

## Afkóðari

hidden3 = tf.layers.dense(hidden1, n_outputs, activation=tf.nn.relu, name="hidden_3")
    
with tf.name_scope("logits"):
    logits = tf.layers.dense(hidden3, n_outputs, name="logits")
    logit = tf.reshape(logits,[-1,81,10], name="2D_logits")

Y_proba = tf.nn.softmax(logit, name="softmax")
##

## Þjálfunar skilgreyning
with tf.name_scope("Training"):
    #reikna loss sem summu cross entropy af reit
    with tf.name_scope("loss"):
        lossArr = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels[:,i], logits=logit[:,i,:])
                for i in range(81)]
        loss = tf.reduce_sum(lossArr)
    
    # minnka learning rate þegar netið er að þjálfast 
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.5, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('Loss', loss) #fyrir tensorboard

##


with tf.name_scope("Solver"):
    val, ind = tf.nn.top_k(logit)
    guess = tf.reshape(ind,[-1,81])


with tf.name_scope("Validate"):
    cellCorr = [tf.nn.in_top_k(logit[:,i,:], labels[:,i], 1) for i in range(81)]
    correct = tf.reduce_all(cellCorr, axis=1) # er öll þrautin rétt?
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    cellacc = tf.reduce_mean(tf.cast(cellCorr, tf.float32))
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Cell_Accuracy', cellacc)


## Dataset
suds = 800000   # Fjöldi þrauta í training
if not 'sudokus' in globals(): # nær ekki aftur í þrautir ef til
    sudokus, _ = getSud(suds)

def getSet(num): 
    rand = np.random.randint(0, suds, num)
    X = sudokus[rand,:]
    return X
##
merged = tf.summary.merge_all() #Sameinna öll summaries
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter("./autocoder/logs/", tf.get_default_graph())
init = tf.global_variables_initializer()

# Keyrslufasi
print('keyrsla')
epochs=100000
batch = 100

with tf.Session() as sess:
    init.run()
    #reuse_saver.restore(sess, "./reuse/save/")
    #saver.restore(sess, "./autocoder/save/")
    for epoch in range(epochs):
        Xtrain = getSet(batch)
        sess.run(training_op, feed_dict={X: Xtrain})
        if epoch%100 == 0:
            summary, gstep = sess.run([merged, global_step],  feed_dict={X: Xtrain})
            file_writer.add_summary(summary, gstep)
            if epoch%1000 == 0:
                Xtrain = getSet(batch)
                loss_test, acc_test, cellacc_test, lrate = sess.run(
                        [loss, accuracy, cellacc, learning_rate],
                        feed_dict={X: Xtrain})

                save_path = saver.save(sess, "./autocoder/save/")
                print("{:8d}:  cell acc: {:4.2f}%  accuracy: {:4.2f}%  Loss: {:4.4}  Learning rate: {:.4f}".format(
                        gstep, cellacc_test*100, acc_test*100, loss_test, lrate))
    summary, gstep = sess.run([merged, global_step],  feed_dict={X: Xtrain})
    file_writer.add_summary(summary, gstep)
    save_path = saver.save(sess, "./autocoder/save/")

    

## prófum netið
with tf.Session() as sess:
    saver.restore(sess,"./autocoder/save/")
    # Vista í reuse til að endurnota
    #reuse_saver.save(sess, "./reuse/save/")
    Xtrain = getSet(100)
    acc_test = accuracy.eval(feed_dict={X: Xtrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    
    # Prentum út eina þraut
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict)
    
file_writer.close()
