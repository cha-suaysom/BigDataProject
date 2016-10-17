import pandas
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

EXPERIMENT_DATA = pickle.load(open('EXPERIMENT_SET_pandas.pkl','rb'))
EVALUATION_SET = pickle.load(open('EVALUATION_SET_pandas.pkl','rb'))
EXPERIMENT_DATA = EXPERIMENT_DATA[EXPERIMENT_DATA["GRAD"] == "YES"]

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
