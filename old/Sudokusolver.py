#Sudoku solver
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def sudata(num_puzzles):
    #Taka inn gogn og skipta nidur i 9x9 fylki
    quizz = np.zeros((num_puzzles, 81), np.int32)
    sol = np.zeros((num_puzzles, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        if i==num_puzzles:
            break
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizz[i, j] = q
            sol[i, j] = s
    return quizz, sol
#Get puzzles and solutions
quizz, sol=sudata(num_puzzles=1000000)
#quizzes = quizzes.reshape((-1, 9, 9))
#solutions = solutions.reshape((-1,9,9))
#Split the data
X_train, X_test, y_train, y_test = train_test_split(quizz,sol,test_size=0.2, random_state=42)
#Configuring the data DNN
config = tf.contrib.learn.RunConfig(tf_random_seed=42)

#feature_cols = [tf.feature_column.numeric_column("x", shape=[1, 81])]
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,300],
                                         n_classes=10,
                                         feature_columns=feature_cols,
                                         config=config,
                                         activation_fn=tf.nn.elu,
                                         optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9))
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)

n=len(X_train)
y_train1=np.zeros(n,np.int32)
i = 0
for i in range(n):
    y_train1[i] = y_train[i,0]
i = 0
dnn_clf.fit(X_train, y_train1, batch_size=100, max_steps=30000)

#dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
#dnn_clf.fit(X_train, y_train, batch_size=81, max_steps=4000)

from sklearn.metrics import accuracy_score
m=len(y_test)
y_pred = dnn_clf.predict(X_test)
y_pred=y_pred['classes']
y_test1=np.zeros(m,np.int32)
i = 0
for i in range(m):
    y_test1[i] = y_test[i,0]

accuracy_score=accuracy_score(y_test1, y_pred)
print(accuracy_score)