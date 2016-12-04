import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
# get_ipython().magic('matplotlib inline')


# experiment_data = pandas.read_excel("data/expSet.xlsx")
# evaluation_set = pandas.read_excel("data/evalSet.xlsx")

EXPERIMENT_DATA = pickle.load(open('EXPERIMENT_SET_pandas.pkl', 'rb'))
EVALUATION_SET = pickle.load(open('EVALUATION_SET_pandas.pkl', 'rb'))
experiment_data = EXPERIMENT_DATA
evaluation_set = EVALUATION_SET
# EXPERIMENT_DATA = EXPERIMENT_DATA[EXPERIMENT_DATA["GRAD"] == "YES"]

# graduated = experiment_data[experiment_data["GRAD"] == "YES"]
graduated = EXPERIMENT_DATA[EXPERIMENT_DATA["GRAD"] == "YES"]


# print("Experiment_Data values {}\n".format(EXPERIMENT_DATA.columns.values))
# print("Evaluation_Set values {}".format(EVALUATION_SET.columns.values))


# print("Shape of Experiment data: {}\nShape of Evaluation Set:{}"
#       .format(experiment_data.shape, evaluation_set.shape))


# EXPERIMENT_DATA.head()


# print("The number of unique locations that we have is {}."
#       .format(len(set(EXPERIMENT_DATA['LOCATION']))))
# print("The number of unique yields that we have is {}."
#       .format(len(set(np.floor(EXPERIMENT_DATA['YIELD'])))))


X = pd.to_numeric(graduated['LOCATION']).reshape(-1, 1)
Y = graduated['YIELD']
knn = KNeighborsRegressor(n_neighbors=100)
knn.fit(X, Y)


predictions = knn.predict(X)


print("Mean Squared Error is: {} ".format(
    np.sum((predictions - Y)**2) / len(Y)))

plt.xlabel("Location")
plt.ylabel("Yield")
plt.title("Location vs. Yield")
plt.scatter(X, Y)
plt.savefig("LocationYieldCluster.png")
# plt.show()
