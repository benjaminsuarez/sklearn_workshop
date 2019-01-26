import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
print(dir(iris))


iris_df = pd.DataFrame(iris.data,columns = iris.feature_names)


ads = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales

boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
boston['MEDV'] = boston_dataset.target


Irish = pd.read_csv("https://raw.githubusercontent.com/benjaminsuarez/sklearn_workshop/master/PPR-ALL.csv")


###

iris = load_iris()

#' each row is an observation, each column is a feature
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

#' in scikit-learn features and response are separate objects
#' lets store feature matrix in X 
#' lets store response vector in y

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

#' scikit-learn is organised into modules, to make it easy to find classes.
import sklearn
dir(sklearn)

#' scikit-learn has a 4 steps to modeling
#' 1: import the class you want to use.
#' 2: instantiate the class. Here you can specify tuning parameters.
#' 3: fit / train the model
#' 4: predict response for a new observation.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X,y)
print(knn.predict([[3,5,4,2]]))

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)
X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
print(knn.predict(X_new))

#' did the model work well?
#' can't  tell with 'out of sample' observations

#' lets examine our training accuracy, ie. the proportion of correct predictions (an evaluation metric for classification problems)
#' lets also compare training accuracy using different values of K.

from sklearn import metrics
print(metrics.accuracy_score(y,y_pred))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y,y_pred))

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X,y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y,y_pred))

#' Our goal is to estimate the likely performance of our model on out-of-sample-data.
#' maximizing training accuracy rewards overfitting the model to the date and the model may not generalize.
#' ie. the model may learn the noise more than the signal.
#' the recommedation is to train-test-split the data...

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#' testing accuracy is a better estimate for out-of-sample-performance then training accuracy!
#' lets repeat what we did previously using our train-test-split.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#' how can we find a better value for K?
#' lets iterate through a list of K values, and plot test accuracys for each value of K.
k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('values of K ')
plt.ylabel('Test Accuracy')
plt.show()

#' once model is chosen, and optimal parameters set, and ready to make predictions using out-of-sample-data
#'remember to retrain the model using all the data. otherwise will be throwing away valuable training data

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X, y) # using all the dataschool
print(knn.predict([[3, 5, 4, 2]])) # an out-of-sample observation

#' # downsides of train-test-split:
#' high variance of out-of-sample accuracy!! accuracy changes a lot depending on what happened to be in the training set.
#' you can use k-fold cross validation instead of train-test-split to overcome this.

#' lets look at advertising data using pandas!!
import pandas as pd

data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col = 0)
print(data.head())
print(data.tail())
print(data.shape)

#' lets use seaborn to visualise the relationships between features and response, pairplots is good for this.
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data, x_vars = ["TV", "radio", "newspaper"], y_vars = "sales", height = 7, aspect = 0.7, kind = "reg")
plt.show()

#' we want to predict sales based on advertising hours, this is a regression problem.
feature_cols = ["TV", "radio", "newspaper"]
X = data[feature_cols] # subset original dataframe
print(X.head)
print(X.shape)
print(type(X))

y = data["sales"] # data.sales also works
print(y.head)
print(y.shape)
print(type(y))

#' split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape) 
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#' default split is 75% for training and 25% for testing

#' Although we'll be using a new class, we follow the same framework for generating our model
#' 1: import the class you want to use.
#' 2: instantiate the class. Here you can specify tuning parameters.
#' 3: fit / train the model
#' 4: predict response for a new observation.

from sklearn.linear_model import LinearRegression

#' instantiate
linreg = LinearRegression()
linreg.fit(X_train, y_train) # learn the coefficients
print(linreg.intercept_) # print the intercept and coefficients
print(linreg.coef_)
print(list(zip(feature_cols, linreg.coef_))) # can pair the feature names with the coefficients

#' making predictions with linear model
y_pred = linreg.predict(X_test)
print(y_pred)

#' we need some evaluation metric to compare our predictions with the actual values. can't use accuracy like in the classification problem before!

#' Mean Absolute Error (MAE)
#' it is the mean of the absolute value of the errors
#' error is the difference between the true and predicted values
#' a short example below:
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
print((10 + 0 + 20 + 10)/4.) # calculate MAE by hand

from sklearn import metrics # calculate MAE using scikit-learn
print(metrics.mean_absolute_error(true, pred))

#' MSE: mean squared error
print((10**2 + 0**2 + 20**2 + 10**2)/4.) # calculate MSE by hand
print(metrics.mean_squared_error(true, pred))
#' MSE is a bit harder to interpret than MAE

#' RMSE: root mean squared error
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.)) # by hand
print(np.sqrt(metrics.mean_squared_error(true, pred)))
#' notice RMSE is a bit larger than MAE, squaring of errors increases the weight of larger errors

#' The bottom line
#' MAE: easiest to understand, it's an average error
#' MSE: more popular than MAE, it punishes larger errors
#' RMSE: more popular than MSE, it is interpretable in the "y" units

#' compute RMSE for sales prediction
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#' value of 1.4 is p.gud as sales ranged from 5 to 25

#' linear reg has no tuning parameters for us to tune.
# can use train_test_split to look at individual features
# lets remove newspapers, as it showed week correlation on visualisation
feature_cols = ['TV', 'radio']
X = data[feature_cols]
y = data.sales
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# RMSE reduces (error is what we want to minimize)
#' it is unlikely this feature is useful for predicting sales and should be removed from the model

#' purpose of model evaluation is to choose the best model: goal is estimate the likely performance of a model on out-of-sample data
#' remember: downside of tran-test-split; high variance estimate of the out-of-sample testing accuracy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

#' lets check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#' there's a difference in test set accuracy!
#' testing accuracy is a high variance estimate like we said
#' we can reduce this variance using cross validation

#' k-fold cross-validation:
#' split data into K folds
#' use 1 fold as testing and remainder as training
#' calculate test accuracy
#' repeat K times with different folds!
#' use average test accuracy as the estimate of out-of-sample accuracy

#' simulation of splitting a dataset of 25 observations into 5 folds using KFold!
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False).split(range(25))

#' print the contents of each training and testing set from simulation
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    print('{:^9} {} {:^25}'.format(iteration, data[0], str(data[1])))

# disadvantages of k cross-validation, difficult to inspect results with confusion matrix or ROC curve
# these are easy to examine with train test split
# recommendations: K=10, for classification problems, stratified sampling  is recommended for creating the folds, each response class should be represented with equal proportions in each of the K folds, "cross-val-score" does this by default

from sklearn.model_selection import cross_val_score
#' 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
#' use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())

#' search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

import matplotlib.pyplot as plt
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# grid_search_cv can do all the above for us!
# k=13 to k=20 seems to work well for KNN, advised to choose simplest model. in KNN higher values of K produce simpler models, as such K=20 would be best.

# lets use cross validation to choose between models!
# 10-fold cross-validation with the best KNN model (after model tuning)
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())
# 10-fold cross-validation with logistic regression (no tuning required)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

# now, lets use cross-validation for feature selection
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales

# 10-fold cross-validation with all three features
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
mse_scores = -scores # fix sign MSE scores as cross_val_score negates it
rmse_scores = np.sqrt(mse_scores) # convert from MSE to RMSE
print(rmse_scores)
print(rmse_scores.mean()) # calculate the average RMSE

# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean())
# results in lower score, which we are trying to minimize, yay!

#' now, lets learn how to use grid search to select the optimal tuning parameters
#' use grid_search_cv for finding optimal K for knn
from sklearn.model_selection import GridSearchCV
#' define the parameter values that should be searched
X = iris.data
y = iris.target
k_range = list(range(1, 31))
print(k_range)
#' create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)
#' instantiate the grid, with our options
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)

#' fit the grid with data, 10 fold cross val is being run 30X times, therofore model is being fit and predictions made 300 times
grid.fit(X, y)
#' view the results as a pandas DataFrame
import matplotlib.pyplot as plt
import pandas as pd
print(pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']])
print(grid.cv_results_)

#' examine the first result
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])
#' print the array of mean scores only
grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)
#' plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
#' examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

#' can use param_grid to map multiple parameters at the same time!!
#' define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']
#' create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)
#' instantiate and fit the grid, this is known as as exhaustive search as all parameters checked
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)
#' view the results
print(pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']])
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)

#' we have successfully tuned our parameters!
#' now lets retrain our model with the best parameters on all the data.
#' GridSearchCV automatically refits best model using all of the data, accessable with .predict
print(grid.predict([[3, 5, 4, 2]]))

#' It can get computationally intensive to perform an exhaustiv tune.
#' RandomizedGridCV solves this
#' it searches only a random subset of the provided parameters. can effectively decide how long you want it to run for
# with RandomizedGridCV you provide a parameter distribution rather than a grid. lets do that.

from sklearn.model_selection import RandomizedSearchCV
param_dist = dict(n_neighbors=k_range, weights=weight_options)
#' n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
rand.fit(X, y)
print(pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']])
print(rand.best_score_)
print(rand.best_params_)
# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
# most of the time it's able to find the best if not closest to the best

best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, return_train_score=False)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)
