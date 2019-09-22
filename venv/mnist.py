import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import operator

#read the data
dataset=pd.read_csv('train.csv')
images=dataset.values[0:,1:]
labels=dataset.values[0:,:1]
X_train, test_images, X_labels, test_labels=train_test_split(images,labels,random_state=2,test_size=0.2)
train_images, valid_images, train_labels, valid_labels = train_test_split(X_train, X_labels, random_state=2,test_size=0.25)

# knn-algorithm
# input: current vector in test_images, train_images, train_labels and k
def classify(inX, dataSet, labels, k):
    # get how many rows in traindata
    dataSetSize = dataSet.shape[0]
    # get the distance
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    seDiffMat = diffMat ** 2
    seDistances = seDiffMat.sum(axis = 1)
    distances = seDistances ** 0.5
    # sort by distance
    sortedDistIndicies = distances.argsort()
    classCount={}
    # get kth shortest distance images
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i],0]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #find the classification that appear mostly
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

start = time.clock()
m = test_labels.shape[0]
resultList = []
errorNum = 0
for i in range(m):
    curResult = classify(test_images[i], train_images, train_labels, 3)
    resultList.append(int(curResult))
    if (int(curResult) != test_labels[i]):
        errorNum += 1.0
    print("\nthe total number of errors is: %d" % errorNum)
    print(i)
    print("######################################")
print("\nthe total error rate is: %f" % (errorNum / float(m)))
cm = confusion_matrix(test_labels, resultList)
print(cm)
end = time.clock()
print ('Time used: {}'.format(end - start))

# def diffK():
#     k_score = []
#     test_range = test_labels.shape[0]
#     for k in range(1, 6 + 1):
#         print("k = {} Training.".format(k))
#         start = time.clock()
#         score = check(k)
#         end = time.clock()
#         k_score.append(score)
#         print("Score: {}.".format(score))
#         print("Complete time: {} Secs.".format(end - start))
#     print(k_score)
#     plt.plot(range(1, 6 + 1), k_score)
#     plt.xlabel('k')
#     plt.ylabel('k_score')
#     plt.show()

# def check(k):
#     m = test_labels.shape[0]
#     resultList = []
#     errorCount = 0
#     for i in range(m):
#         nowLabel = classifyDigit(test_images[i], train_images, train_labels, k)
#         resultList.append(int(nowLabel))
#         print ("the classifier came back with: %d, the real answer is: %d" % (int(nowLabel), test_labels[i]))
#         if (int(nowLabel) != test_labels[i]):
#             errorCount += 1.0
#         print ("\nthe total number of errors is: %d" % errorCount)
#         print(i)
#         print("######################################")
#     print("\nthe total error rate is: %f" % (errorCount / float(m)))
#     cm = confusion_matrix(test_labels, resultList)
#     print(cm)


#
# if __name__ == '__main__':
#     start = time.clock()
#     print(images.shape)
#     print(labels.shape)
#     end = time.clock()
#     print ('Time used: {}'.format(end - start))