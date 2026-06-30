import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("BankNote_Authentication.csv")
print(df.describe())
print(df.isnull().sum())
print(df.info())

print(df["class"].unique())

x=df.drop("class",axis=1)
y=df['class']

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) 

model=KNeighborsClassifier()

param_grid={
    'n_neighbors':[3,5,7,9],
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan','minkowski']
}

grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='accuracy')

grid.fit(x_train,y_train)
prediction=grid.predict(x_test)



print("Best Parameters:",grid.best_params_)
print("Cross Validation Score:",grid.best_score_)
print("Confusion Matrix:\n",confusion_matrix(y_test,prediction))
print("Classification Report:\n",classification_report(y_test,prediction))  
print("Accuracy Score:",accuracy_score(y_test,prediction)) 

sns.pairplot(df,hue="class")
plt.show()

sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.show()

sns.countplot(x="class",data=df)
plt.show()
