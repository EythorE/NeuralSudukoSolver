#Sudoku solver
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
quizz, sol=sudata(num_puzzles=100)
#quizzes = quizzes.reshape((-1, 9, 9))
#solutions = solutions.reshape((-1,9,9))
#Split the data
X_train, X_test, y_train, y_test = train_test_split(quizz,sol,test_size=0.2, random_state=42)
#Configuring the data DNN
config = tf.contrib.learn.RunConfig(tf_random_seed=42)
#feature_cols = [tf.feature_column.numeric_column("x", shape=[1, 81])]
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100],
n_classes=81,
feature_columns=feature_cols,
config=config)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
clfs = [dnn_clf for i in range(81)]

print('Er að fitta:')
i = 0
for clf in tqdm(clfs):
    clf.fit(X_train, y_train[:,i], batch_size=81, max_steps=4000)
    i=i+1
#dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
#dnn_clf.fit(X_train, y_train, batch_size=81, max_steps=4000)

#from sklearn.metrics import accuracy_score
#y_pred = dnn_clf.predict(X_test)
#accuracy_score(y_test, y_pred['classes'])

sud2pred = np.zeros((1, 81), np.int32)

Sud2 = X_test[:1]
Sud2sol = y_test[:1]

i = 0
for clf in clfs:
    if Sud2[0,i] == 0:  
        sud2pred[0,i] = clf.predict(sum(Sud2))
    else:
        sud2pred[0,i] = Sud2[0,i]
    i=i+1

test=sud2pred==Sud2sol
number=np.sum(test)
m,n=test.shape
accuracy=100*number/(m*n)
print("Nakvaemni: %.4f" % accuracy + '%')
