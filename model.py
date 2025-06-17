import pandas as pd
from sklearn.pipeline import Pipeline

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

pipeline.fit(X_train,Y_train)
predictions = pipeline.predict(X_test) 

scores = pipeline.score(X_test,Y_test)
print("Accuracy of model:", scores)

from sklearn.metrics import precision_score,recall_score
precision = precision_score(Y_test,predictions)
recall = recall_score(Y_test,predictions)
print('Precision:', precision)
print('Recall:', recall)

import pickle
pickle.dump(pipeline, open('pipeline.pkl', 'wb'))

print(dataset.columns.tolist())