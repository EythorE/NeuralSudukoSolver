# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:38:10 2018

@author: eytor
"""
import numpy as np
def getSuduko(numPuzzles):
    quizzes = np.zeros((numPuzzles, 81), np.int32)
    solutions = np.zeros((numPuzzles, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        if i==numPuzzles:
            break
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    return quizzes, solutions

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

numPuzzles=10000
X,y = getSuduko(numPuzzles)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

clfs = [ExtraTreesClassifier() for i in range(81)]

from tqdm import tqdm


print('Er að fitta:')
i = 0
for clf in tqdm(clfs):
    clf.fit(X_train, y_train[:,i])
    i=i+1

sud2pred = np.zeros((1, 81), np.int32)

Sud2 = X_test[:1]
Sud2sol = y_test[:1]

i = 0
for clf in clfs:
    if Sud2[0,i] == 0:  
        sud2pred[0,i] = clf.predict(Sud2)
    else:
        sud2pred[0,i] = Sud2[0,i]
    i=i+1
sudokupred = sud2pred.reshape((-1, 9, 9))
sudokusol = Sud2sol.reshape((-1, 9, 9))

test=sud2pred==Sud2sol
number=np.sum(test)
m,n=test.shape
accuracy=100*number/(m*n)
print("Nakvaemni: %.4f" % accuracy + '%')


#
#y_test_pred = clf.predict(X_test)
#
#accuracy = accuracy_score(y_test,y_test_pred)
#
#print('Accuracy:')
#print(accuracy)
#print()

sudokupred = sud2pred.reshape((-1, 9, 9))
sudokusol = Sud2sol.reshape((-1, 9, 9))
