# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:34:00 2018

@author: eytor
"""
import numpy as np

def checkNumber(sudoku,i,j,t):
    for r in range(9):
        if sudoku[r,j] == t:
            return False
    for c in range(9):
        if sudoku[i,c] == t:
            return False
    rbox = int(i/3)
    cbox = int(j/3)
    for n in range(3):
        for m in range(3):
            if sudoku[rbox*3+n,cbox*3+m] == t:
                return False
    return True
    

#solves for only possible numbers in rows,columns, and boxes
#returns True if solved, else false
def solveObvious(sudoku):
    changedNum = True
    #first = True
    al=0
    while changedNum:
        al=al+1
        changedNum = False
        canChange = sudoku == 0;
        if np.all(canChange==False):
            return True #Búinn að leysa
        for i in range(9):
            for j in range(9):
                if canChange[i,j]:
                    count = 0;
                    for t in range(1,10):
                        if checkNumber(sudoku,i,j,t) == True:
                            num = t
                            count = count + 1
                            if count > 1:
                                break
                    if count == 1:
                        sudoku[i,j] = num
                        changedNum = True
                        #first = False
    #return first #tölu var breytt
        print(al)
        print(sudoku)
    return False  #Ekki búinn að leysa                 

#returns solution and number of solved puzzles
def solver(puzzles):
    counter = 0
    sudokus = puzzles.reshape((-1, 9, 9))
    for sudoku in sudokus:
        if solveObvious(sudoku):
            counter = counter+1
    sudokus = sudokus.reshape((-1, 81))
    return sudokus, counter

#returns solution and indexes of unsolved solutions
    #includes proccess bar
from tqdm import tqdm
def solver2(puzzles):
    unsolvedIndex = []
    sudokus = puzzles.reshape((-1, 9, 9))
    for i, sudoku in enumerate(tqdm(sudokus)):
        if not solveObvious(sudoku):
            unsolvedIndex.append(i)

    sudokus = sudokus.reshape((-1, 81))
    return sudokus, unsolvedIndex




#print('Er að fitta:')
#i = 0
#for clf in tqdm(clfs):
#    clf.fit(X_train, y_train[:,i])
#    i=i+1
app = np.array([0,7,9,0,0,0,0,0,8,0,0,0,1,4,0,2,0,0,0,0,0,0,0,9,0,0,4,9,0,0,0,7,8,0,0,0,0,6,0,0,9,0,0,4,0,0,8,0,0,0,0,0,0,3,0,5,1,8,6,0,0,0,0,0,0,8,0,0,0,0,5,2,0,0,0,5,0,0,0,0,0])
appsol = app.reshape(9,9)
print(appsol)
s = solveObvious(appsol)

#sudoku = [ [0,0,4, 3,0,0, 2,0,9], [0,0,5, 0,0,9, 0,0,1], [0,7,0, 0,6,0, 0,4,3], [0,0,6, 0,0,2, 0,8,7], [1,9,0, 0,0,7, 4,0,0], [0,5,0, 0,8,3, 0,0,0], [6,0,0, 0,0,0, 1,0,5], [0,0,3, 5,0,8, 6,9,0], [0,4,2, 9,1,0, 3,0,0] ]
#sudoku = np.array(sudoku)
#solveObvious(sudoku)