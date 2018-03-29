import tensorflow as tf
import numpy as np

tf.reset_default_graph() 

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

def data(num):
    Xd = np.random.randint(0,2,(num,10))
    yd = np.ones((num,10),dtype=Xd.dtype) - Xd
    return Xd,yd

with tf.Session() as sess:
    init.run()
    reuse_saver.restore(sess, "./save_test/notgate")
    Xt, yt = data(100)
    l = out.eval(feed_dict={X: Xt})
    print("Reload Sucess:")
    print(np.all(l==yt)) #skilar true ef allt var rétt

 