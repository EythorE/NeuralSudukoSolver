#Kodi fyrir Sudoku leysara
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
from sklearn import model_selection

def sudata(num_puzzles):
    #Taka inn gogn og skipta nidur i 9x9 fylki
    quizzes = np.zeros((num_puzzles, 81), np.int32)
    solutions = np.zeros((num_puzzles, 81), np.int32)
    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        if i==num_puzzles:
            break
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
    return quizzes, solutions
#quizzes1 = quizzes.reshape((-1, 9, 9))
#solutions1 = solutions.reshape((-1, 9, 9))
quizzes, solutions=sudata(num_puzzles=1000)
#Skipta nidur gognunum
X_train, X_test, y_train, y_test = model_selection.train_test_split(quizzes,solutions,test_size=0.2, random_state=42)

startRNDM = time.time()#Byrja ad taka tima fyrir Random forest
RNDM_FRST=RandomForestClassifier(random_state=42)
RNDM_FRST.fit(X_train,y_train)
endRNDM = time.time()
timeRNDM=endRNDM-startRNDM

y_predFRST=np.int32(RNDM_FRST.predict(X_test))
print("Random: %.4f s" % timeRNDM)

test=y_predFRST==y_test
number=np.sum(test)
m,n=test.shape
accuracy=100*number/(m*n)
print("Nakvaemni: %.4f" % accuracy + '%')


