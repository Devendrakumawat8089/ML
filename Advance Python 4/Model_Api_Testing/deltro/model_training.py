import numpy as np 
import pandas as pd 

df =pd.read_csv("Social_Network_Ads.csv")
# print(df.head(2)) 

df = df.drop(columns = ['User ID' , 'Gender'])

x = df.drop(columns = ['Purchased'])
y = df['Purchased'] 

from sklearn.model_selection import train_test_split 
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier() 
rf.fit(x_train , y_train) 

import joblib 

# Save the model
joblib.dump(rf, 'rf_model.pkl')
print("Model saved as rf_model.pkl")