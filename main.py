import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.drop_duplicates

bins = [0,2000,3000,4000,10000]
labels = ["light","medium","Heavy","Very Heavy"]
df["weight_bin"] = pd.cut(df["weights"],bins=bins,labels=labels,include_lowest=True)
