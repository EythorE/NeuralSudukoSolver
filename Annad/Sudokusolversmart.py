#Sudoku solver
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import copy
from keras import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score


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
quizz, sol=sudata(num_puzzles=100000)
#Split the data
X_train1, X_test1, y_train1, y_test1 = train_test_split(quizz,sol,test_size=0.2, random_state=42)

X_train1 = np.array([np.reshape([int(d) for d in flatten_grid], (81))
                      for flatten_grid in X_train1])
y_train1 = np.array([np.reshape([int(d) for d in flatten_grid], (81))
                      for flatten_grid in y_train1])
X_test1 = np.array([np.reshape([int(d) for d in flatten_grid], (81))
                      for flatten_grid in X_test1])
y_test1 = np.array([np.reshape([int(d) for d in flatten_grid], (81))
                      for flatten_grid in y_test1])

X_train = to_categorical(y_train1).astype('float32')  # from ytrain cause we will creates quizzes from solusions
X_test = to_categorical(X_test1).astype('float32')

y_train = to_categorical(y_train1-1).astype('float32') # (y - 1) because we 
y_test = to_categorical(y_test1-1).astype('float32')   # don't want to predict zeros

input_shape = (81, 10)

model=Sequential()
model.add(Dense(units=64,activation='relu',input_shape=input_shape))
model.add(Flatten())
nesterov=optimizers.SGD(momentum=0.9, nesterov=True)

grid = Input(shape=input_shape)  # inputs
features = model(grid)  # commons features

digit_placeholders = [Dense(9, activation='softmax')(features)
    for i in range(81)]

solver=Model(grid,digit_placeholders)
solver.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

solver.fit(X_train,
    [y_train[:,i,:] for i in range(81)],  # each digit of solution
    batch_size=50,
    epochs=1, # 1 epoch should be enough for the task
    verbose=1) 

prediction=solver.predict(X_test)
pred=np.zeros([20000,81],np.int32)
for i in range(81):
    for j in range(20000):
        m=max(prediction[i][j])
        pred[j,i]=sum([k for k, l in enumerate(prediction[i][j]) if l == m])+1

y_sol=np.zeros(81,np.int32)
i = 0
for i in range(81):
    y_sol[i] = y_test1[0,i]


#score=accuracy_score(y_test1, pred)
#print(score)
test=np.zeros([20000,81],np.int32)
for i in range(20000):
    for j in range(81):
        test[i,j]=y_test1[i,j]==pred[i,j]
number=np.sum(test)
m,n=test.shape
accuracy=100*number/(m*n)
print("Nakvaemni: %.4f" % accuracy + '%')




