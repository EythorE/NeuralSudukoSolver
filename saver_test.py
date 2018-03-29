import tensorflow as tf
import numpy as np
tf.reset_default_graph() 

X = tf.placeholder(tf.float32, shape=(None, 10), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="y")

layer = tf.layers.dense(X, 20, activation=tf.nn.relu, name="layer")
logits = tf.layers.dense(layer, 10, name="logits")

# Næ í breyturnar sem ég vill save-a og bý til saver fyrir þær
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
reuse_saver = tf.train.Saver(reuse_vars_dict)

# óþarfi að save-a það sem kemur hér á eftir
y_true = tf.nn.sigmoid(logits)
cmp = tf.greater(y_true, 0.5) # returns boolean tensor
out = tf.cast(cmp, tf.int32) # casts boolean tensor into int32


Xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_sum(Xentropy)

learning_rate = 0.0001
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


def data(num):
    Xd = np.random.randint(0,2,(num,10))
    yd = np.ones((num,10),dtype=Xd.dtype) - Xd
    return Xd,yd

with tf.Session() as sess:
    init.run()
    for epoch in range(10000):
        Xt, yt = data(100)
        sess.run(training_op, feed_dict={X: Xt, y: yt})
    Xt, yt = data(100)
    l = out.eval(feed_dict={X: Xt})
    print("Train Success:")
    print(np.all(l==yt)) #skilar true ef allt var rétt
    save_path = reuse_saver.save(sess, "./save_test/notgate")
    




tf.reset_default_graph() # graph tómt eins og að byrja frá byrjun

X = tf.placeholder(tf.float32, shape=(None, 10), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="y")
layer = tf.layers.dense(X, 20, activation=tf.nn.relu, name="layer")
logits = tf.layers.dense(layer, 10, name="logits")

# Næ í breyturnar sem ég vill loada og bý til saver fyrir þær
reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
reuse_saver = tf.train.Saver(reuse_vars_dict)

y_true = tf.nn.sigmoid(logits)
cmp = tf.greater(y_true, 0.5) # returns boolean tensor
out = tf.cast(cmp, tf.int32) # casts boolean tensor into int32
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    reuse_saver.restore(sess, save_path)
    Xt, yt = data(100)
    l = out.eval(feed_dict={X: Xt})
    print("Reload Sucess:")
    print(np.all(l==yt)) #skilar true ef allt var rétt

 