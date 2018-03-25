import numpy as np
import matplotlib.pyplot as plt

def getSud(numPuzzles):
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


def plotSud(X, y=None, ycorr=None ):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0,9)
    ax.set_ylim(9,0)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(3.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(3.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0))
    ax.grid(which='major', axis='x', linewidth=4, linestyle='-', color='black')
    ax.grid(which='minor', axis='x', linewidth=1, linestyle='-', color='black')
    ax.grid(which='major', axis='y', linewidth=4, linestyle='-', color='black')
    ax.grid(which='minor', axis='y', linewidth=1, linestyle='-', color='black')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.set_size_inches(8, 8)
    for i in range(0,9):
        for j in range(0,9):
            if(X[0,j+i*9] != 0):
                ax.text(j+0.5, i+0.5, str(X[0,j+i*9]),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=30, color='black', fontweight='bold')
            if(y is not None):
                if(y[0,j+i*9] != X[0,j+i*9]):
                    ax.text(j+0.5, i+0.5, str(y[0,j+i*9]),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=30, color='blue', fontweight='normal')
            if(ycorr is not None):
                if(y[0,j+i*9] != ycorr[0,j+i*9] and X[0,j+i*9] == 0):
                    ax.text(j+0.8, i+0.8, str(ycorr[0,j+i*9]),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=20, color='red', fontweight='normal')
    
    # savefig('../figures/grid_ex.png',dpi=48)
    plt.show()



if __name__ == "__main__":
    X,ycorr = getSud(1)
    plotSud(X)
    y = np.copy(ycorr)
    np.random.seed(42)
    for i in range(15):
        randIndex = np.random.randint(0,len(y[0,:])-1)
        randNum = np.random.randint(0,9)
        y[0,randIndex] = randNum
    plotSud(X,y ,ycorr)


