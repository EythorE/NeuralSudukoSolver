# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 01:21:57 2018

@author: eytor
"""

from solveObvious import solver,solver2
import numpy as np

def getSudoku(numPuzzles):
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

numb2solve = 100
X, y = getSudoku(numb2solve)
y_sol, unsolved = solver2(X)
print(unsolved)

print(np.all(y_sol==y))

#skoðum óleystu
uns = []
for i in unsolved:
    uns.append(y_sol[i])
uns=np.array(uns)
uns =  uns.reshape((-1, 9, 9))
print('percent solved:')
print((numb2solve-uns.shape[0])/numb2solve*100)