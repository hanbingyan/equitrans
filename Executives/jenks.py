# code from http://danieljlewis.org/files/2010/06/Jenks.pdf
# described at http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
import numpy as np


def getJenksBreaks(dataList_, numClass):
    dataList = dataList_.copy()
    dataList.sort()
    mat1 = []
    for i in range(0, len(dataList) + 1):
        temp = []
        for j in range(0, numClass + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(0, len(dataList) + 1):
        temp = []
        for j in range(0, numClass + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, numClass + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(dataList) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(dataList) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(dataList[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, numClass + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(dataList)
    kclass = []
    for i in range(0, numClass + 1):
        kclass.append(0)
    kclass[numClass] = float(dataList[len(dataList) - 1])
    countNum = numClass
    while countNum >= 2:  # print "rank = " + str(mat1[k][countNum])
        id = int((mat1[k][countNum]) - 2)
        # print "val = " + str(dataList[id])
        kclass[countNum - 1] = dataList[id]
        k = int((mat1[k][countNum] - 1))
        countNum -= 1
    return kclass


def getGVF(dataList_, numClass):
    """
    The Goodness of Variance Fit (GVF) is found by taking the
    difference between the squared deviations
    from the array mean (SDAM) and the squared deviations from the
    class means (SDCM), and dividing by the SDAM
    """
    dataList = dataList_.copy()
    breaks = getJenksBreaks(dataList, numClass)
    dataList.sort()
    listMean = sum(dataList) / len(dataList)
    # print(listMean)
    SDAM = 0.0
    for i in range(0, len(dataList)):
        sqDev = (dataList[i] - listMean) ** 2
        SDAM += sqDev
    SDCM = 0.0
    for i in range(0, numClass):
        if breaks[i] == 0:
            classStart = 0
        else:
            classStart = dataList.index(breaks[i])
            classStart += 1
        classEnd = dataList.index(breaks[i + 1])
        classList = dataList[classStart:classEnd + 1]
        classMean = sum(classList) / len(classList)
        # print(classMean)
        preSDCM = 0.0
        for j in range(0, len(classList)):
            sqDev2 = (classList[j] - classMean) ** 2
            preSDCM += sqDev2
        SDCM += preSDCM
    return (SDAM - SDCM) / SDAM


# written by Drew
# used after running getJenksBreaks()
def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value <= breaks[i]:
            return i
    return len(breaks) - 1

# a = [0.61407712, 0.52192977, 0.58054592, 0.47592635, 0.52145686,
#      0.84245183, 0.57345875, 0.58177329, 0.46289373, 1.        ,
#        0.67990388, 0.55449764, 0.60727295, 0.71735531, 0.54878393,
#        0.57922013, 0.55324935, 0.48872387, 0.44300624, 0.63686544,
#        0.52766578, 0.47920201, 0.61751363, 0.44995187, 0.47743878,
#        0.53491503, 0.6716771 , 0.55699843, 0.85817325, 0.53125847,
#        0.45994587, 0.47484841, 0.50011842, 0.62233057, 0.49351523,
#        0.60196367, 0.61303992, 0.52246365, 0.62886424, 0.52423658,
#        0.81390725, 0.62262651, 0.55089512, 0.58448684, 0.49073091,
#        0.47403374, 0.48816169, 0.49801532, 0.52424117, 0.52111336,
#        0.54655606, 0.5344884 , 0.5557082 , 0.5147456 , 0.68121155,
#        0.54006852, 0.72881414, 0.50277271, 0.36726411, 0.36114368,
#        0.36442189, 0.52513097, 0.55222404, 0.64272016, 0.44323962,
#        0.50966324, 0.48555079, 0.51686993, 0.44859161, 0.53412516,
#        0.51075498, 0.43092586, 0.55397132, 0.57652057, 0.85999272,
#        0.48587182, 0.4928455 , 0.53626845, 0.52004511, 0.59358616,
#        0.77309046, 0.47606036, 0.53615167, 0.52731267, 0.58831418,
#        0.46496063]
#
# # a = np.arange(1, 41, 1).tolist()
# breaks = getJenksBreaks(a, 4)
# print(a)
# print(classify(39, breaks))
# # print(len(a))
# # for num in range(1, 100):
# #     print('num', num, getGVF(a, num))