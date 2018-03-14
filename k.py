import numpy as np
import operator
from os import listdir

def handwritingClassTest():
    #样本数据标签列表
    hwLabels = []

    #样本数据文件列表
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)

    #初始化样本数据矩阵
    trainingMat = np.zeros((m, 1024))

    #依次读取所有样本数据到数据矩阵
    for i in range(m):
        #提取文件名中的数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        #将样本数据存入矩阵
        trainingMat[i:] = img2Vector('digits/trainingDigits/%s' % fileNameStr)

    #循环读取测试数据
    testFileList = listdir('digits/testDigits/')

    #初始化错误率
    errorCount = 0.0
    mTest = len(testFileList)

    #循环测试每个测试文件数据
    for i in range(mTest):
        #提取文件中的数字
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        #提取数据向量
        vectorUnderTest = img2Vector('digits/testDigits/%s' % fileNameStr)

        #对数据文件分类
        classifileResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print("the classifier came back with: %d, the real answer is: %d" % (classifileResult, classNumStr))
        
        #判断KNN算法是否准确
        if(classifileResult != classNumStr):
            errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total errors rate is: %f" % (errorCount/float(mTest)))

        

def img2Vector(filename):
    #创建向量
    Vect = np.zeros((1,1024))

    #打开数据文件，读取每行内容
    fr = open(filename)

    for i in range(32):
        #读取每一行
        lineStr = fr.readline()
        #将每一行前32字符转变成int存入向量
        for j in range(32):
            Vect[0, 32*i+j] = int(lineStr[j])

    return Vect

#testVector = img2Vector('digits/testDigits/0_1.txt')
#print(testVector[0, 0:31])
#inX：用于分类的输入向量
#dataSet：输入的训练样本集
#labels：样本数据的类标签向量
#k：用于选择最近邻居的数目
def classify0(intX, dataSet, labels, k):
    #获取样本数据数量
    dataSetSize = dataSet.shape[0]

    #矩阵运算，计算测试数据与样本数据对应数据项的差值
    diffMat = np.tile(intX, (dataSetSize,1)) - dataSet

    #上一步结果平方和
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)

    #取平方跟，得到距离向量
    distance = sqDistance**5

    #按照距离从低到高排序
    sortedDistance = distance.argsort()
    classCount = {}

    #依次取出最近的样本数据
    for i in range(k):
        #记录该样本数据所属的类别
        votellabel = labels[sortedDistance[i]]
        classCount[votellabel] = classCount.get(votellabel, 0) + 1

    #对类别出现的频率进行排序，从高到低
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    
    #返回出现频率最高的类别
    return sortedClassCount[0][0]

handwritingClassTest()
