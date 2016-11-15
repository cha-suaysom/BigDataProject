import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

EXPERIMENT_DATA = pickle.load(open('EXPERIMENT_SET_pandas.pkl','rb'))
EVALUATION_SET = pickle.load(open('EVALUATION_SET_pandas.pkl','rb'))
EXPERIMENT_DATA = EXPERIMENT_DATA[EXPERIMENT_DATA["GRAD"] == "YES"]
EXPERIMENT_DATA = EXPERIMENT_DATA[["RM","YIELD","BAGSOLD"]]
def obtainw(l, X, Y):
	dataSize = X.shape[1]
	I_D = np.identity(dataSize)
	I_D[dataSize-1,dataSize-1] = 0 
	answer = l*I_D + np.dot(X.T,X)
	answer = answer.astype(float)
	answer = np.linalg.inv(answer)
	answer = np.dot(answer,X.T)
	answer = np.dot(answer,Y)
	return answer

#data_news = pandas.read_csv('online_news_popularity.csv')
data_news = EXPERIMENT_DATA.astype(np.float)
data_news["BAGSOLD"] = np.log(data_news["BAGSOLD"])
fractionTraining = 1/2
fractionValidating = 1
dataSize = len(data_news)

trainingDataSize = int(fractionTraining * dataSize)
testingDataSize = dataSize - trainingDataSize
validationSetSize = int(fractionValidating * testingDataSize)
myTestSetSize = testingDataSize - validationSetSize

training_data_news = data_news.head(trainingDataSize)
testing_data_news = data_news.tail(testingDataSize)
validation_set_data_news = testing_data_news.head(validationSetSize)
test_set_data_news = testing_data_news.tail(myTestSetSize)

#The x_i 
featureSize = 2
outputSize = 1
matrix_training_data_news = training_data_news.as_matrix()
X_training = matrix_training_data_news[:,0:(featureSize)]
X_training = np.c_[X_training, np.ones(X_training.shape[0])]
Y_training = matrix_training_data_news[:,-outputSize]
#Y_training = np.log(Y_training.astype(float))

matrix_validation_data_news = validation_set_data_news.as_matrix()
X_validation = matrix_validation_data_news[:,0:(featureSize)]
#add a bias term
X_validation = np.c_[X_validation, np.ones(X_validation.shape[0])]
Y_validation = matrix_validation_data_news[:,-outputSize]
#Y_validation = np.log(Y_validation.astype(float))

matrix_set_data_news = test_set_data_news.as_matrix()
X_test = matrix_set_data_news[:,0:(featureSize)]
X_test = np.c_[X_test, np.ones(X_test.shape[0])]
Y_test = matrix_set_data_news[:,-outputSize]
#Y_test = np.log(Y_test.astype(float))

do = 1
if (do == 1):
	numLambda = 500 #number of different lambda to try
	randomNumber = numLambda*2.235*1e2*(np.random.random_sample(numLambda))
	randomNumber.sort()
	#randomNumber = pickle.load(open('randomNumber.pkl','rb'))
	WtoSave = np.zeros((numLambda, featureSize + 1))
	pickle.load(open('randomNumber.pkl','rb'))
	
	""" Already done"""
	done = 0
	if (done!=1):
		for i in range(len(randomNumber)):
			print(randomNumber[i])
			WtoSave[i] = obtainw(randomNumber[i],X_training,Y_training)

		
		pickle.dump(WtoSave, open('wMatrixTrain.pkl','wb'))

	wMatrix = pickle.load(open('wMatrixTrain.pkl','rb'))
	MSEList = []
	for l in range(len(randomNumber)):
		MSE = np.dot(X_validation, wMatrix[l].T)-Y_validation
		MSE = MSE**2
		MSE = np.sum(MSE)
		MSE = (MSE/len(Y_validation))**(1/2)
		MSEList.append(MSE)
#print(MSEList)
plt.plot(randomNumber,MSEList)
plt.xlabel('Lambda (Regularization Term)')
plt.ylabel('RMSE on validation set')
plt.title('RMSE on validation set vs lambda')
plt.show()

def plotRegression():
	bagSold = np.asarray(EXPERIMENT_DATA["BAGSOLD"]).reshape(-1,1).astype(np.float)
	rm = np.asarray(EXPERIMENT_DATA["RM"]).reshape(-1,1).astype(np.float)
	plt.subplot(121)
	plt.title('Linear Regression RM versus Bagsold')
	plt.plot(rm,bagSold,'ro')
	regr = linear_model.LinearRegression()
	regr.fit(rm,bagSold)
	plt.plot(rm,regr.predict(rm),color = 'red')

	plt.subplot(122)
	plt.title('Linear Regression RM versus log of Bagsold')
	bagSold = np.log(bagSold)
	plt.plot(rm,bagSold,'ro')
	regr = linear_model.LinearRegression()
	regr.fit(rm,bagSold)
	plt.plot(rm,regr.predict(rm),color = 'red')

	plt.show()
