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

n_hidden1 = 81*14
n_hidden2 = 81*7

n_hidden3 = n_hidden1

n_cells = 81    # fjöldi reita í þraut
cell_nums = 10     # fjöldi talna til að giska á (10(0-9) eða 9(1-9))
n_outputs = n_cells*cell_nums

X = tf.placeholder(tf.uint8, shape=(None, 81), name="X")
OH = tf.one_hot(X, 10, dtype=tf.float32)
inp = tf.layers.flatten(OH)

sol = tf.placeholder(tf.int32, shape=(None, 81), name="Solutions")
# dreg frá einn, þá er labels frá 0 uppí 8
# nema núll sé með, þá er labels frá 0 uppí 9
y = tf.add(sol , tf.constant(cell_nums-10,sol.dtype), name= "y") 


## Nota autocoder lög
reuse=tf.AUTO_REUSE
hidden1 = tf.layers.dense(inp, n_hidden1, activation=tf.nn.relu,
                          trainable=False, name="hidden_1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                          trainable=False, name="hidden_2")

# næ í breyturnar sem ég vill save-a og bý til saver fyrir þær
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
reuse_saver = tf.train.Saver(reuse_vars_dict, name="Sjalfkodari")


## Skilgreyning á tauganeti # skoða hvort við þurfum eitthvað regularization
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu,
                          name="hidden_3")
#hidden4 = tf.layers.dense(hidden3, n_outputs, activation=tf.nn.relu, name="hidden_4")
#hidden5 = tf.layers.dense(hidden4, n_outputs, activation=tf.nn.relu, name="hidden_5")

    
with tf.name_scope("logits"):
    logits = tf.layers.dense(hidden3, n_outputs, name="logits")
    logit = tf.reshape(logits,[-1,n_cells,cell_nums], name="2D_logits")
    



##

## Þjálfunar skilgreyning
with tf.name_scope("Training"):
    #reikna loss sem summu cross entropy af reit
    with tf.name_scope("Cross-entropy"):
        lossArr = [
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y[:,i], logits=logit[:,i,:])
                for i in range(n_cells)]
    loss = tf.reduce_sum(lossArr)

        
    # minnka learning rate þegar netið er að þjálfast 
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.9, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('Loss', loss) #fyrir tensorboard
##


with tf.name_scope("Solver"):
    val, ind = tf.nn.top_k(logit)
    guess8 = tf.reshape(ind,[-1,81])
    guess = tf.add(guess8, tf.constant(10-cell_nums,dtype=guess8.dtype))
    y_proba = tf.nn.softmax(logit, name="y_proba__softmax__")


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
if not 'solutions' in globals(): # nær ekki aftur í þrautir ef til
    sudokus,solutions = getSud(suds)

def getSet(num, maxempty=0): #removes rand[0:maxempty] numbers
    rand = np.random.randint(0, suds, num)

#    y = solutions[rand,:]

#    X = np.copy(y) # Lausnir sem þrautir
#    for i in range(num):
#        empty = np.random.randint(0, maxempty+1) # how many to remove
#        X0ind = np.random.randint(0, 81, empty) # indexes set to 0
#        X[i,X0ind]=0
    
    X = sudokus[rand,:] # Þrautir
    y = np.copy(X)
    return X,y
##

merged = tf.summary.merge_all() #Sameinna öll summaries
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter("./sudoku_fitter/logs/",
                                    tf.get_default_graph())
init = tf.global_variables_initializer()


# Keyrslufasi
print('keyrsla')
epochs=20000
batch = 100
rm = 0          #max empty cells

with tf.Session() as sess:
    init.run()
    
    # Bara nota eitt af þessum
    # Nota reuse aðeins fyrst og skipta svo
    # Loadar vigtum fyrir lag 1 og 2 frá sjálfkóðara
    reuse_saver.restore(sess, "./reuse/save/") 
    #saver.restore(sess, "./sudoku_fitter/save/") #Loada öllum breytum

    for epoch in range(epochs):
        Xtrain, ytrain = getSet(batch,rm)
        if epoch%100 == 0:
            summary, gstep = sess.run([merged, global_step],
                                      feed_dict={X: Xtrain, sol: ytrain})
            file_writer.add_summary(summary, gstep)
            if epoch%10000 == 0:
                save_path = saver.save(sess, "./sudoku_fitter/save/")
                loss_test, acc_test, cellacc_test, lrate = sess.run(
                        [loss, accuracy, cellacc, learning_rate],
                        feed_dict={X: Xtrain, sol: ytrain})
                print("{:8d}:  cell acc: {:4.2f}%  accuracy: {:4.2f}%  Loss: {:4.4}  Learning rate: {:.4f}".format(
                        gstep, cellacc_test*100, acc_test*100, loss_test, lrate))
        sess.run(training_op, feed_dict={X: Xtrain, sol: ytrain})
            
    
    # final save
    save_path = saver.save(sess, "./sudoku_fitter/save/")
    summary, gstep = sess.run([merged, global_step],
                               feed_dict={X: Xtrain, sol: ytrain})
    file_writer.add_summary(summary, gstep)
    
    # Validate
    Xtrain, ytrain = getSet(100,rm)
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    if cell_nums == 9:
        plotSud(Xtrain[:1,:], y=ypredict, ycorr=ytrain[:1,:])
    if cell_nums == 10:
        plotSud(Xtrain[:1,:], y=ypredict)

## prófum netið
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./sudoku_fitter/save/")
    Xtrain, ytrain = getSet(100,rm)
    acc_test = accuracy.eval(feed_dict={X: Xtrain, sol: ytrain})
    print("Loaded test accuracy: {:.2f}%".format(acc_test * 100))
    
    # Prentum út eina þraut
    ypredict = guess.eval(feed_dict={X: Xtrain[:1,:]})
    fig = plotSud(Xtrain[:1,:], y=ypredict)
#    

file_writer.close()