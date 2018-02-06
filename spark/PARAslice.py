#####################################################################
#####################################################################
#
# 			PARAslice-Spark
#
# A Parallel Factor Analysis (PARAFAC) Alternating Least Squares (ALS)
# tensor decomposition implementation in Apache Spark for dense, skewed
# tensors with 3 modes. Optionally supports regularization and/or sketching.
#
#    Kareem S. Aggour <aggour@ge.com>
#
# NOTE: tensor dimensions are of the form [z,x,y] NOT [x,y,z]!!!!!
#
#####################################################################
#####################################################################

import numpy as np
import ast
import argparse
import datetime
import math
import subprocess
from lib import tensorOps
from pyspark import SparkContext
from tensorly.kruskal import kruskal_to_tensor
from tensorly.tenalg import norm
from tensorly.tenalg import khatri_rao
from numpy.linalg import solve
from tensorly.base import unfold, fold
from pyspark.accumulators import AccumulatorParam

from operator import add
import gc
import pyspark

sc = SparkContext(appName='PARAFAC in Spark')

#####################################################################
# input variable default values
#####################################################################
# Tensor rank, i.e., number of components in decomposition
R = 5

# stopping criteria
maxIter = 500
minErrDelta = 1e-05
numExec = 1000

randomSeed = 0

# input & output directory
inputDir='/user/aggour/rpi/spark/tensor-500x500x5x10000/'
outputDir=''

# Regularization in case problem is ill-conditioned
regularization = 0  # None
#regularization = 1  # L2 (Tikhonov aka Ridge regression)
#regularization = 2  # Proximal regularization

# regularization parameter (can't call it lambda in Spark!)
# references on selecting regularization parameter
#   http://researchrepository.murdoch.edu.au/id/eprint/4407/1/parameter_choice_methods.pdf
#   http://www2.compute.dtu.dk/~pcha/DIP/chap5.pdf
regulParam = 0.001

# Sketching
sketching = 0 # No sketching
#sketching = 1 # CPRAND
#sketching = 2 # random slice selection
#sketching = 3 # random entry selection

# Amount of slices or entries to select when sketching
sketchingRate = 0.000001

#####################################################################
# global variables
#####################################################################
# which mode of the decomposition we are currently solving (A, B, or C)
decompMode = 0

# RxR identity matrix used for regularization
eye = 0

# tensor dimensions (will be calculated from data, just need global variables set to 0)
I = 0
J = 0
K = 0

# A and B decomposition matrices (Ci matrices will be local to partitions)
A = 0
B = 0

# sketching row selection
sketchingRowsA = []
sketchingRowsB = []
sketchingRowsC = []

#####################################################################
# functions
#####################################################################
def getMSDiff(diff):
    """
    Return time difference in milliseconds
    """
    ms = diff.days*24*60*60*1000.0
    ms += diff.seconds * 1000.0
    ms += 1.0 * diff.microseconds / 1000.0
    return ms

def getMS(start, stop):
    """
    Get time difference in milliseconds
    """
    diff = stop - start
    return getMSDiff(diff)

def saveFactorMatrices(partition):
    """
    Spark job to solve for and save each Ci factor matrix.
    """
    ret = []
    rows = list(partition)
    error = 0.0
    for row in rows:
        label = row[0]
        Xi = row[1]
        Ki = Xi.shape[0]
	dashIdx=label.rindex('-')
	dotIdx=label.rindex('.')
	labelId=int(label[dashIdx+1:dotIdx])

	# solve for Ci
	Ci = np.zeros((Ki,R))
	ZiTZic = tensorOps.ZTZ(A, B)
	XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
	if regularization > 0:
	    ZiTZic = ZiTZic + regulParam * eye
	Ci = solve(ZiTZic.T, XiZic.T).T
	#print Ci

	if outputDir!='':
	    # save Ci
	    filename = './Ci-' + str(labelId)
	    np.save(filename, Ci)

	    # save A & B
	    if labelId==0:
		filename = './A'
		np.save(filename, A)
		filename = './B'
		np.save(filename, B)

	error = error + np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))

    if outputDir!='':
	subprocess.call(['hadoop fs -moveFromLocal ' + './*.npy ' + outputDir], shell=True)

    ret.append(['error',error])
    return ret

def getTensorDimensions(partition):
    """
    Spark job to process each slice and return its local tensor dimensions.
    """
#    print '****** get tensor dim ******'
    ret = []
    rows = list(partition)
    for row in rows:
	Xi = row[1]
	a = []
	a.extend(Xi.shape)
	a.append(np.square(norm(Xi, 2)))
	ret.append(a)
    return [tensorOps.getDim (ret)]

def initializeData(partition):
    """
    Converts binary file of tensor slice to in-memory numpy array.  Input 
    partitions are of the form (tensor_slice_filename, binary content) 
    and outputs are of the form (tensor_slice_filename, numpy array of tensor slice).
    """
#    print '******* initializing *******'
    ret = []
    rows = list(partition)
    for row in rows:
	fsav = file('tmp.npy', 'wb')
	fsav.write(row[1])
	fsav.close()
	Xi = np.load('tmp.npy')
        ret.append([row[0], Xi])
    return ret

def singleModeALSstep(partition):
    """
    Runs a single step of Alternating Least Squares to solve for one of A (mode = 1),
    B (mode = 2), or C (mode = 3) matrix.
    """
    '''
    if decompMode == 1:
        print 'Solving for A....'
    elif decompMode == 2:
        print 'Solving for B....'
    elif decompMode == 3:
        print 'Solving for Ci...'
    '''
    ret = []
    rows = list(partition)
    ZiTZi = 0
    XiZi = 0

    error = 0.0

    for row in rows:
        label = row[0]
        Xi = row[1]
        Ki = Xi.shape[0]
	if sketching > 0:
	    dashIdx=label.rindex('-')
	    dotIdx=label.rindex('.')
	    labelId=int(label[dashIdx+1:dotIdx])
	    minIndex = labelId
	    maxIndex = labelId + Ki - 1
	    selectRowsC = sketchingRowsC[(sketchingRowsC >= minIndex) & (sketchingRowsC <= maxIndex)]
	    selectRowsC = selectRowsC - minIndex
	    if len(selectRowsC) == 0:
		continue;

	# always solve for Ci first!
	Ci = np.zeros((Ki,R))
	if sketching == 1 or sketching == 3:
            ZiTZic = tensorOps.ZTZ(A[sketchingRowsA,:], B[sketchingRowsB,:])
            XiZic = np.dot(unfold(Xi[:,sketchingRowsA,:][:,:,sketchingRowsB], 0), khatri_rao([Ci, A[sketchingRowsA,:], B[sketchingRowsB,:]], skip_matrix=0))
	# don't need a sketching == 2, since else is the same
	else:
            ZiTZic = tensorOps.ZTZ(A, B)
            XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
        #ZiTZic = tensorOps.ZTZ(A, B)
        #XiZic = np.dot(unfold(Xi, 0), khatri_rao([Ci, A, B], skip_matrix=0))
        if regularization > 0:
            ZiTZic = ZiTZic + regulParam * eye
	# I don't have Ci yet...
	#if regularization == 2:
	#    XiZi = XiZi + regulParam * Ci
        Ci = solve(ZiTZic.T, XiZic.T).T

        if decompMode == 1:
	    if sketching == 1 or sketching == 3:
                ZiTZi = ZiTZi + tensorOps.ZTZ(B[sketchingRowsB,:], Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:][:,:,sketchingRowsB], 1), khatri_rao([Ci[selectRowsC,:], A, B[sketchingRowsB,:]], skip_matrix=1))
	    elif sketching == 2:
                ZiTZi = ZiTZi + tensorOps.ZTZ(B, Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:], 1), khatri_rao([Ci[selectRowsC,:], A, B], skip_matrix=1))
	    else:
                ZiTZi = ZiTZi + tensorOps.ZTZ(B, Ci)
#                XiZi = XiZi + tensorOps.unfolded_3D_matrix_multiply(decompMode, Xi, Ci, B, I, J, Ki, R)
                XiZi = XiZi + np.dot(unfold(Xi, 1), khatri_rao([Ci, A, B], skip_matrix=1))
        elif decompMode == 2:
	    if sketching == 1 or sketching == 3:
                ZiTZi = ZiTZi + tensorOps.ZTZ(A[sketchingRowsA,:], Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:][:,sketchingRowsA,:], 2), khatri_rao([Ci[selectRowsC,:], A[sketchingRowsA,:], B], skip_matrix=2))
	    elif sketching == 2:
                ZiTZi = ZiTZi + tensorOps.ZTZ(A, Ci[selectRowsC,:])
                XiZi = XiZi + np.dot(unfold(Xi[selectRowsC,:,:], 2), khatri_rao([Ci[selectRowsC,:], A, B], skip_matrix=2))
	    else:
                ZiTZi = ZiTZi + tensorOps.ZTZ(A, Ci)
#                XiZi = XiZi + tensorOps.unfolded_3D_matrix_multiply(decompMode, Xi, Ci, A, I, J, Ki, R)
                XiZi = XiZi + np.dot(unfold(Xi, 2), khatri_rao([Ci, A, B], skip_matrix=2))
        elif decompMode == 3:
	    if sketching == 1 or sketching == 3:
		error = error + np.square(norm(Xi[selectRowsC,:,:][:,sketchingRowsA,:][:,:,sketchingRowsB] - kruskal_to_tensor([Ci[selectRowsC,:], A[sketchingRowsA,:], B[sketchingRowsB,:]]), 2))
	    elif sketching == 2:
		error = error + np.square(norm(Xi[selectRowsC,:,:] - kruskal_to_tensor([Ci[selectRowsC,:], A, B]), 2))
	    else:
		error = error + np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))
        else:
            print 'Unknown decomposition mode. Catastrophic error. Failing now...'

    if (len(rows) > 0) and (decompMode < 3):
        ret.append(['ZTZ',ZiTZi])
        ret.append(['XZ',XiZi])
    elif (decompMode == 3):
        ret.append(['error',error])
    del ZiTZi, XiZi
    return ret

def calculateFNorm(partition):
    """
    Calculate Frobenius Norm of tensor slices.
    """
    ret = []
    rows = list(partition)
    normX = 0.0
    for row in rows:
        Xi = row[1]
        normX = normX + np.square(norm(Xi, 2))
        '''
        (Ki,I,J) = Xi.shape
        for i in range(0,I):
            for j in range(0,J):
                for k in range(0,Ki):
                    normX = normX + np.square(Xi.item((k,i,j)))
        '''
    return ([normX])

def calculateError(partition):
    """
    Calculate Frobenius Norm of difference between tensor slices and decomposed tensor.
    """
    ret = []
    rows = list(partition)
    normX = 0.0
    error = 0.0
    for row in rows:
        Xi = row[1]
        Ci = row[2]
        normX = normX + np.square(norm(Xi, 2))
        error = error + np.square(norm(Xi - kruskal_to_tensor([Ci, A, B]), 2))
        '''
        (Ki,I,J) = Xi.shape
        for i in range(0,I):
            for j in range(0,J):
                for k in range(0,Ki):
                    sum = 0.0
                    for r in range(0,R):
                        sum = sum + A.item(i,r) * B.item(j,r) * Ci.item(k,r)
                    x = Xi.item((k,i,j))
                    error = error + np.square(sum) - (2.0*sum*x)
                    normX = normX + np.square(x)
        '''
    ret.append(['error',error])
    ret.append(['normX',normX])
    return ret

def parafac_als(inputDir, outputDir, numExec, R, maxIter, minErrDelta, regularization, regulParam, sketching, randomSeed):
    """
    Run PARAFAC ALS on tensor in input directory.
    :param inputDir:
      Input directory of tensor slice files.
    :param outputDir:
      Output directory of factor matrix slice files.
    :param numExec:
      Number of Spark executors to use. If numExec is less than the number of slices, not all may be used.
    :param R:
      Number of rank-1 tensors to decompose input tensor into.
    :param maxIter:
      Stopping criteria - maximum number of iterations.
    :param minErrDelta:
      Stopping criteria - minimum error delta after which we stop.
    :param regularization:
      Type of regularization (0 = None, 1 = L2/Tikhonov, 2 = Proximal)
    :param regulParam:
      Regularization parameter (a.k.a., lambda)
    :param sketching:
      Sketching approach (0 = None, 1 = CPRAND, 2 = % rows, 3 = % of entries)
    """
    global decompMode
    global I, J, K
    global A, B
    global eye
    global sketchingRowsC
    global sketchingRowsA
    global sketchingRowsB
    global sketchingRate

    print '********************************************************************'
    print '********************************************************************'
    print '********************************************************************'
    print 'Initializing...'
    print '    Input directory:', inputDir
    if outputDir!='':
	print '    Output directory:', outputDir
    print '    Number of Spark executors:', numExec
    print '    Tensor rank:', R
    print '    Stopping criteria:'
    print '        Max iterations:',maxIter
    print '        Min error delta:',minErrDelta
    if regularization==0:
        print '    Regularization: None'
    elif regularization==1:
        print '    Regularization: L2/Tikhonov'
        print '        Regularization parameter:',regulParam
    elif regularization==2:
        print '    Regularization: Proximal'
        print '        Regularization parameter:',regulParam
    if sketching==0:
	print '    Sketching: None'
    elif sketching==1:
	print '    Sketching: CPRAND'
    elif sketching==2:
	print '    Sketching: Random slice selection'
	print '    Sketching rate:',sketchingRate
    elif sketching==3:
	print '    Sketching: Random entry selection'
	print '    Sketching rate:',sketchingRate

    startAll = datetime.datetime.now()
#    print 'Reading files from HDFS', datetime.datetime.now()
    rows = sc.binaryFiles(inputDir, numExec)

    # turn tensor binary files into ndarray's
#    tensorRDD = rows.mapPartitions(initializeData).persist(pyspark.StorageLevel.MEMORY_AND_DISK)
#    tensorRDD = rows.mapPartitions(initializeData).persist(pyspark.StorageLevel.DISK_ONLY)
    tensorRDD = rows.mapPartitions(initializeData).persist(pyspark.StorageLevel.MEMORY_ONLY)
    del rows
    print '    Tensor slice count =', tensorRDD.count()

    # Do a first pass to get dimensions of slices to initialize A and B
    # also get the Frobenius norm of the tensor
    print 'Getting tensor dimensions...'
#    print 'Getting tensor dimensions', datetime.datetime.now()
    dimRDD = tensorRDD.mapPartitions(getTensorDimensions).collect()
    (K,I,J,normX) = tensorOps.getDim (dimRDD)
    I = int(I)
    J = int(J)
    K = int(K)
    eye = np.identity (R)
    print '    I =',I,', J =',J,', K =',K
    print '       normX^2 =',normX

    # set random seed so I can use the same initial conditions across runs
    if randomSeed != 0:
	np.random.seed(randomSeed)

    print 'Initializing decomposition matrices...'
#    print 'Initializing decomposition matrices', datetime.datetime.now()
    # initialize A
    A = np.random.rand(I,R)
    #print 'A:\n--\n',A
    # initialize B
    B = np.random.rand(J,R)
    #print 'B:\n--\n',B

    # set max values for skewed sketching
    if sketching==1:
	sketchingRows = math.ceil(10 * R * math.log(R, 10))
	sketchingRows_square_root = int(math.ceil(math.sqrt(sketchingRows)))
	sketchingRows_P = 372.0
	sketchingRows_P_cube_root = int(math.ceil(sketchingRows_P**(1.0/3)))
	sketchingRows_IJK = (1.0 * I * J * K) / sketchingRows_P
	sketchingRate = sketchingRows / (I*J*K*1.0)
    elif sketching==2:
	sketchingRows = int(math.ceil(K * sketchingRate))
    elif sketching==3:
	sketchingRows = math.ceil(I * J * K * sketchingRate)
	sketchingRows_square_root = int(math.ceil(math.sqrt(sketchingRows)))
	sketchingRows_P = sketchingRows
	sketchingRows_P_cube_root = int(math.ceil(sketchingRows_P**(1.0/3)))
	sketchingRows_IJK = (1.0 * I * J * K) / sketchingRows_P

    # initialize error values and stopping criterion thresholds
    oldError = 10
    lowestError = 10
    errorNoChangeCount = 0
    errorChangeBelowThreshold = 0
    maxItersWithoutErrorChange = 30
    maxItersWithErrorChangeBelowThreshold = 5
    if sketchingRate > 0:
	maxItersWithoutErrorChange = round(4.0361*pow(sketchingRate,-0.107))
#	maxItersWithoutErrorChange = round(3.1533*pow(sketchingRate,-0.11))
    errDelta = 1

    # PARAFAC Alternating Least Squares loop
    print 'Executing decomposition...'
#    print 'Executing decomposition', datetime.datetime.now()
    mapTime = 0.0
    errorTime = 0.0
    startSteps = datetime.datetime.now()
    #maxExecTime = 0.0
    for step in range(0,maxIter):
#        print '--- Iteration',step
        # solve for A
        decompMode = 1
#        print 'Solving for A', datetime.datetime.now()
	if sketching==1 or sketching==3:
	    numB = np.random.randint(2,min(sketchingRows_square_root, J))
	    numC = int(math.ceil(sketchingRows/numB))
	    sketchingRowsA = np.unique(np.random.randint(0,I,numC))
	    sketchingRowsB = np.unique(np.random.randint(0,J,numB))
	    sketchingRowsC = np.unique(np.random.randint(0,K,numC))
	elif sketching==2:
	    sketchingRowsC = np.random.randint(0,K,sketchingRows)
	startMap = datetime.datetime.now()
        XZandZTZ = tensorRDD.mapPartitions(singleModeALSstep)
        sums = XZandZTZ.reduceByKeyLocally(add)
	mapTime = mapTime + getMS(startMap, datetime.datetime.now())
        XZ = sums['XZ']
        ZTZ = sums['ZTZ']
        if regularization > 0:
            ZTZ = ZTZ + regulParam * eye
        if regularization == 2:
            XZ = XZ + regulParam * A
#        A = np.matmul(XZ, np.linalg.inv(ZTZ))
        A = solve(ZTZ.T, XZ.T).T
#        print 'new A=\n',A
        del XZandZTZ, XZ, ZTZ

        # solve for B
        decompMode = 2
#        print 'Solving for B', datetime.datetime.now()
	if sketching==1 or sketching==3:
	    numA = np.random.randint(2,min(sketchingRows_square_root, I))
	    numC = int(math.ceil(sketchingRows/numA))
	    sketchingRowsA = np.unique(np.random.randint(0,I,numA))
	    sketchingRowsB = np.unique(np.random.randint(0,J,numC))
	    sketchingRowsC = np.unique(np.random.randint(0,K,numC))
	startMap = datetime.datetime.now()
        XZandZTZ = tensorRDD.mapPartitions(singleModeALSstep)
        # 'Locally' automatically creates a dict of the results
        sums = XZandZTZ.reduceByKeyLocally(add)
	mapTime = mapTime + getMS(startMap, datetime.datetime.now())
        XZ = sums['XZ']
        ZTZ = sums['ZTZ']
        if regularization > 0:
            ZTZ = ZTZ + regulParam * eye
        if regularization == 2:
            XZ = XZ + regulParam * B
#        B = np.matmul(XZ, np.linalg.inv(ZTZ))
        B = solve(ZTZ.T, XZ.T).T
#        print 'new B=\n',B
        del XZandZTZ, XZ, ZTZ

        # solve for Ci's
        decompMode = 3
#        print 'Solving for C', datetime.datetime.now()
	if sketching==1 or sketching==3:
	    numA = np.random.randint(2,min(sketchingRows_P_cube_root, I))
	    numB = np.random.randint(2,min(sketchingRows_P_cube_root, J))
	    numC = int(math.ceil(sketchingRows_P/(numA*numB)))
	    sketchingRowsA = np.unique(np.random.randint(0,I,numA))
	    sketchingRowsB = np.unique(np.random.randint(0,J,numB))
	    sketchingRowsC = np.unique(np.random.randint(0,K,numC))
	startMap = datetime.datetime.now()
        errorRDD = tensorRDD.mapPartitions(singleModeALSstep)

        # calculate error
#        print 'Calculating error', datetime.datetime.now()
	sums = errorRDD.reduceByKeyLocally(add)
	mapTime = mapTime + getMS(startMap, datetime.datetime.now())
	errorTime = errorTime + getMS(startMap, datetime.datetime.now())

	if sketching==1 or sketching==3:
	    newError = np.sqrt((sketchingRows_IJK * sums['error']) / normX)
	elif sketching==2:
	    newError = np.sqrt(sums['error'] / (sketchingRate * normX))
	else:
            newError = np.sqrt(sums['error'] / normX)
        del errorRDD, sums
	errDelta = oldError - newError
	if sketching > 0:
	    if errDelta > 0.0 and newError < lowestError:
		lowestError = newError
		errorNoChangeCount = 0
	    else:
		errorNoChangeCount = errorNoChangeCount + 1
	    if errDelta > 0.0 and errDelta < minErrDelta:
		errorChangeBelowThreshold = errorChangeBelowThreshold + 1
	    else:
		errorChangeBelowThreshold = 0
	    errDelta = 1 # need to reset this when sketching so it doesn't cause loop to stop
        print 'Iteration',step, ' error =',newError,' delta =',errDelta

	'''
        # calculate full error at each step
        errorRDD = tensorRDD.mapPartitions(saveFactorMatrices)
        sums = errorRDD.reduceByKeyLocally(add)
        newError = np.sqrt(sums['error'] / normX)
        print '   curr err =',newError
	'''

	if newError == 1.0:
	    print '      Sums =',sums
	    print 'A:\n',A
	    print 'B:\n',B
	if errDelta < minErrDelta or errorNoChangeCount >= maxItersWithoutErrorChange or errorChangeBelowThreshold >= maxItersWithErrorChangeBelowThreshold:
	    break
	oldError = newError

    endAll = datetime.datetime.now()

    # calculate the final true error and save the factor matrices
    errorRDD = tensorRDD.mapPartitions(saveFactorMatrices)
    sums = errorRDD.reduceByKeyLocally(add)
    newError = np.sqrt(sums['error'] / normX)

    '''
    print '\n\nFinal Results:\n--------------'
    print 'A:\n--\n',A
    print 'B:\n--\n',B
    '''

    sc.stop ()

    print ''
    print 'Run summary\n-----------'
    print 'Input directory:', inputDir
    if outputDir!='':
	print 'Output directory:', outputDir
    print 'Input tensor dimensions:'
    print '    I =',I,', J =',J,', K =',K
    print '    # entries in tensor:',I*J*K
    print 'Tensor rank:', R
    if regularization==0:
        print 'Regularization: None'
    elif regularization==1:
        print 'Regularization: L2/Tikhonov'
        print '    Regularization parameter:',regulParam
    elif regularization==2:
        print 'Regularization: Proximal'
        print '    Regularization parameter:',regulParam
    if sketching==0:
	print 'Sketching: None'
    elif sketching==1:
	print 'Sketching: CPRAND'
    elif sketching==2:
	print 'Sketching: Random slice selection'
	print 'Sketching rate:',sketchingRate
    elif sketching==3:
	print 'Sketching: Random entry selection'
	print 'Sketching rate:',sketchingRate
    print 'Final error:',newError
    step = step + 1
    print 'Number of iterations:',step
    totalRuntime = getMS(startAll, endAll) / 1000.0
    totalSteptime = getMS(startSteps, endAll) / 1000.0
    mapTime = mapTime / 1000.0
    errorTime = errorTime / 1000.0
    print 'Total runtime (sec):',totalRuntime
    print 'Average runtime (sec):','{0:.6f}'.format(totalRuntime/step)
    print 'Total map time (sec):',mapTime
    print 'Average map time (sec):','{0:.6f}'.format(mapTime/step)
    print 'Total error calc time (sec):',errorTime
    print 'Average error calc time (sec):','{0:.6f}'.format(errorTime/step)
    print 'Total steptime (sec):',totalSteptime
    print 'Average steptime (sec):','{0:.6f}'.format(totalSteptime/step)

    print '\nKSA,',inputDir,',',I,',',J,',',K,',',R,',',regularization,',',regulParam,',',sketching,',',sketchingRate,',',step,',',newError,',',totalRuntime,',',(totalRuntime/step),',',totalSteptime,',',(totalSteptime/step),',',mapTime,',',(mapTime/step),',',errorTime,',',(errorTime/step)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Spark-based implementation of PARAFAC ALS.')
    parser.add_argument('-I', '--inputDir', help='Input directory (hdfs or local)', type=str, required=False, default=inputDir)
    parser.add_argument('-O', '--outputDir', help='Output directory (hdfs or local)', type=str, required=False, default=outputDir)
    parser.add_argument('-R', '--r', help='Tensor rank or number of components in decomposition', type=int, required=False, default=R)
    parser.add_argument('-M', '--maxIter', help='Maximum number of iterations (stopping criteria)', type=int, required=False, default=maxIter)
    parser.add_argument('-D', '--minErrDelta', help='Minimum error delta (stopping criteria)', type=float, required=False, default=minErrDelta)
    parser.add_argument('-G', '--regularization', help='Regularization type: 0 = None, 1 = L2/Tikhonov, 2 = Proximal', type=int, required=False, default=regularization)
    parser.add_argument('-S', '--sketching', help='Sketching mode: 0 = None, 1 = CPRAND, 2 = Slice sampling by rate, 3 = Entry sampling by rate', type=int, required=False, default=sketching)
    parser.add_argument('-K', '--sketchingRate', help='Sketching rate (e.g., 0.1 for 10%)', type=float, required=False, default=0)
    parser.add_argument('-L', '--regulParam', help='Regularization parameter (lambda)', type=float, required=False, default=regulParam)
    parser.add_argument('-E', '--numExecutors', help='Number of Spark executors', type=int, required=False, default=numExec)
    parser.add_argument('-Sd', '--seed', help='Random seed', type=int, required=False, default=0)
    args = parser.parse_args()
#    print (args)
    R = args.r
    inputDir = args.inputDir
    outputDir = args.outputDir
    maxIter = args.maxIter
    minErrDelta = args.minErrDelta
    regularization = args.regularization
    regulParam  = args.regulParam
    sketching = args.sketching
    sketchingRate = args.sketchingRate
    numExec = args.numExecutors
    randomSeed = args.seed

    # Make sure the output directory exists and is empty
    if outputDir!='':
	subprocess.call(['hadoop fs -rm -r -skipTrash ' + outputDir], shell=True)
	subprocess.call(['hadoop fs -mkdir ' + outputDir], shell=True)
	subprocess.call(['hadoop fs -chmod 777 ' + outputDir], shell=True)

    parafac_als(inputDir, outputDir, numExec, R, maxIter, minErrDelta, regularization, regulParam, sketching, randomSeed)


