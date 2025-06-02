import numpy as np
import pandas as pd

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#Tried regularization no changes in accuracy, precision and recall

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# I had to scale the data because the model didn't fully converge and stopped before reaching the best solution

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_scaled,Y, test_size=0.1,random_state=0)

model.fit(X_train,Y_train)
predictions = model.predict(X_test) 
#print("Predictions:", predictions)

scores = model.score(X_test,Y_test)
print("Accuracy of model:", scores)

from sklearn.metrics import precision_score,recall_score
precision = precision_score(Y_test,predictions)
recall = recall_score(Y_test,predictions)
print('Precision:', precision)
print('Recall:', recall)

print("Enter the details asked below to check your heart's state!")
age = int(input("Enter your age(in years):"))
sex = int(input("Your sex(0=female,1=male):"))
chestPainType = int(input("Type of chest pain(1 = typical angina,2 = atypical angina,3 = non-anginal pain,4 = asymptomatic):"))
bloodPressure = int(input("Resting blood pressure(in mm):"))
serumCholestrol = int(input("Cholestrol(in mg/dl):"))
bloodSugar = int(input("Fasting Blood Sugar(1 = sugar > 120mg/dl, 0 = sugar < 120mg/dl):"))
restingECG = int(input("Resting ECG(0 = normal, 1 = ST-T wave abnormality, 2 = Probable or Definite Left Ventricular hypertrophy):"))
maxHeartRate = int(input("Max Heart Rate:"))
exercisedAngina = int(input("Exercise Angina(0 = no, 1 = yes):"))
oldpeak = float(input("Oldpeak:"))
Stslope = int(input("ST Slope(1 = upward,2 = flat,3 = downward):"))

user_input = pd.DataFrame([[age,sex,chestPainType,bloodPressure,
                        serumCholestrol,bloodSugar,restingECG,
                        maxHeartRate,exercisedAngina,oldpeak,Stslope]], 
                        columns=X.columns)

user_input_scaled = scaler.transform(user_input)

user_prediction = model.predict(user_input_scaled)
print("\nPrediction Result:", "No Heart Disease" if user_prediction[0] == 1 else "Has a Heart Disease")
#print("Coefficient of the Model:", model.coef_)
#print("Intercept of the Model:", model.intercept_)