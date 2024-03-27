import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit as sss
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
titanic_data =pd.read_csv('train.csv')
print(titanic_data)
print(titanic_data.head())
print(titanic_data.describe())
print(titanic_data.info())
numeric_columns = titanic_data.select_dtypes(include=[np.number])

#Plot the heatmap
sns.heatmap(numeric_columns.corr(),cmap="RdGy")
plt.show()
split=sss(n_splits=1,test_size=0.2)
for train , test in split.split(titanic_data,titanic_data[['Survived','Pclass','Sex']]):
    start_train_set=titanic_data.loc[train]
    start_test_set=titanic_data.loc[test]
plt.subplot(1,2,1)
start_train_set['Survived'].hist()
start_train_set["Pclass"].hist()
plt.subplot(1,2,2)
start_test_set['Survived'].hist()
start_test_set["Pclass"].hist()
plt.show()
class AgeImputer(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        imputer=SimpleImputer(strategy="mean")
        X['Age']=imputer.fit_transform(X[['Age']])
        return X


class Feauture_Encoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        encoder = OneHotEncoder()
        embarked_matrix = encoder.fit_transform(X[["Embarked"]]).toarray()
        embarked_columns = encoder.get_feature_names_out(["Embarked"])
        embarked_df = pd.DataFrame(embarked_matrix, columns=embarked_columns)

        sex_matrix = encoder.fit_transform(X[["Sex"]]).toarray()
        sex_columns = encoder.get_feature_names_out(["Sex"])
        sex_df = pd.DataFrame(sex_matrix, columns=sex_columns)

        return pd.concat([X, embarked_df, sex_df], axis=1).drop(["Embarked", "Sex"], axis=1)



class Feauture_Dropper(BaseEstimator,TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        return X.drop(["Embarked_nan","Embarked","Name","Sex","Ticket","Cabin","N"],axis=1,errors="ignore")
pipeline = Pipeline([
    ("ageimputer", AgeImputer()),  # No quotes around AgeImputer
    ("Feautureencoder", Feauture_Encoder()),
    ("Feauturedropper", Feauture_Dropper())
])
imputer = SimpleImputer(strategy='median')
start_train_set=pipeline.fit_transform(start_train_set)
print(start_train_set)
pd.set_option('display.max_columns', None)
start_train_set.info()
X=start_train_set.drop(['Survived'],axis=1)
y=start_train_set['Survived']
scaler=StandardScaler()
X_data=scaler.fit_transform(X)
y_data=y.to_numpy()
print(X_data)
print(y_data)
y_data_imputed = imputer.fit_transform(y_data.reshape(-1, 1)).ravel()
clf=RandomForestClassifier()
param_grid=[{"n_estimators":[10,20,50],"max_depth":[None,5,10],"min_samples_split":[2,3]  }]
grid_search=GridSearchCV(clf,param_grid,cv=3,scoring="accuracy",return_train_score=True)
grid_search.fit(X_data,y_data_imputed)
final_clf=grid_search.best_estimator_
print(final_clf)
print(final_clf.score(X_data,y_data_imputed))
start_test_set=pipeline.fit_transform(start_test_set)
X_test=start_test_set.drop(["Survived"],axis=1)
y_test=start_test_set["Survived"]
scaler=StandardScaler()
X_data_test=scaler.fit_transform(X_test)
y_data_test=y_test.to_numpy()
print(y_data_test)
y_data_test_imputed = imputer.fit_transform(y_data_test.reshape(-1, 1)).ravel()
print(y_data_test_imputed)
# print(final_clf.score(X_data_test,y_data_test_imputed))
print(X_data_test)
final_data=pipeline.fit_transform(titanic_data)
X_final=final_data.drop(["Survived"],axis=1)
y_final=final_data["Survived"]
scaler=StandardScaler()
X_data_final=scaler.fit_transform(X_final)
y_data_final=y_final.to_numpy()
y_data_final_imputed = imputer.fit_transform(y_data_final.reshape(-1, 1)).ravel()
final_clf.score(X_data_test,y_data_test_imputed)
final_clf=RandomForestClassifier()
param_grid=[{"n_estimators":[10,20,50],"max_depth":[None,5,10],"min_samples_split":[2,3]  }]
grid_search=GridSearchCV(final_clf,param_grid,cv=3,scoring="accuracy",return_train_score=True)
grid_search.fit(X_data_final,y_data_final_imputed)
prod_final_clf=grid_search.best_estimator_
print(prod_final_clf)
titanic_test_data=pd.read_csv("test.csv")
final_test_data=pipeline.fit_transform(titanic_test_data)
final_test_data.info()
X_final_test=final_test_data
scaler=StandardScaler()
X_data_final_test=scaler.fit_transform(X_final_test)
predictions=prod_final_clf.predict(X_data_final_test)
predictions = predictions.astype(int)
final_df=pd.DataFrame(titanic_test_data["PassengerId"])
final_df["Survived"]=predictions
final_df.to_csv("predictions.csv",index=False)

