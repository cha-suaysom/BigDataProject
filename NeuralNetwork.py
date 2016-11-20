import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

EXPERIMENT_DATA = pickle.load(open('EXPERIMENT_SET_pandas.pkl','rb'))
EVALUATION_SET = pickle.load(open('EVALUATION_SET_pandas.pkl','rb'))
EXPERIMENT_DATA["GRAD"] = EXPERIMENT_DATA["GRAD"].replace(["YES", "NO"], [1,0])
EXPERIMENT_DATA["CHECK"] = EXPERIMENT_DATA["CHECK"].astype(int)
EXPERIMENT_DATA = EXPERIMENT_DATA[EXPERIMENT_DATA["GRAD"] == 1]
EXPERIMENT_DATA = EXPERIMENT_DATA[["RM","BAGSOLD"]]

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
featureSize = 1
outputSize = 1
matrix_training_data_news = training_data_news.as_matrix()
X_training = matrix_training_data_news[:,0:(featureSize)]
Y_training = matrix_training_data_news[:,-outputSize]


matrix_validation_data_news = validation_set_data_news.as_matrix()
X_validation = matrix_validation_data_news[:,0:(featureSize)]
#add a bias term
Y_validation = matrix_validation_data_news[:,-outputSize]

matrix_set_data_news = test_set_data_news.as_matrix()
X_test = matrix_set_data_news[:,0:(featureSize)]
Y_test = matrix_set_data_news[:,-outputSize]

clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(10,100,100), random_state=1)

clf.fit(X_training, Y_training)
result = clf.predict(X_validation)
MSE = result - Y_validation
MSE = MSE**2
MSE = np.sum(MSE)
MSE = (MSE/len(Y_validation))**(1/2)
print("MSE is: ", MSE)