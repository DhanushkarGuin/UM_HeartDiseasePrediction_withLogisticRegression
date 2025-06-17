import pickle
import pandas as pd

pipeline = pickle.load(open('pipeline.pkl', 'rb'))

columns= ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina', 'oldpeak', 'ST slope']

test_input = pd.DataFrame([[32,1,2,120,240,0,0,170,1,1.5,1]], columns = columns)

predictions = pipeline.predict(test_input)
print(predictions)