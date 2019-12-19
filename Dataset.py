
#IMPORTING THE MODULES
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#LOADING THE DATA

#takes the csv file from the gitbhub and loads it into the program and saves it
#into the variable dataset with the columns of names
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length","sepal-width","petal-length","petal-width","class"]
dataset = read_csv(url, names=names)


#SHOWING THE DATA

#shows the rows and columns of the dataset (150,5)
print(dataset.shape)
print()
print()

#shows the first 20 lines of data
print(dataset.head(20))
print()
print()

#shows some numerics for the data such as the count, mean, min and max values
print(dataset.describe())
print()
print()

#shows the number of instances that belong to each class (e.g. Iris-setosa 50)
print(dataset.groupby("class").size())
print()
print()

