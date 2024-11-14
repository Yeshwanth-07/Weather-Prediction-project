import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('data.csv')

from itertools import chain
def Create_List(x):
  list_of_lists = [w.split() for w in x.split(',')]
  flat_list = list(chain(*list_of_lists))
  return flat_list



def Get_weather(list1):
  if 'Rain' in list1 and 'Fog' in list1:
    return 'RAIN+FOG'
  elif 'Snow' in list1 and 'Rain' in list1:
    return 'RAIN+SNOW'
  elif 'Rain' in list1:
    return 'RAIN'
  elif 'Snow' in list1:
    return 'SNOW'
  elif 'Fog' in list1:
    return 'FOG'
  elif 'Clear' in list1:
    return 'CLEAR'
  elif 'Cloudy' in list1:
    return 'CLOUDY'
  else:
    return 'RAIN'

data['Std_Weather'] = data['Weather'].apply(lambda x: Get_weather(Create_List(x)))

cloudy_df = data[data['Std_Weather'] == 'CLOUDY'].sample(600)

clear_df = data[data['Std_Weather'] == 'CLEAR'].sample(600)

rain_df = data[data['Std_Weather'] == 'RAIN']

snow_df = data[data['Std_Weather'] == 'SNOW']

weather_df = pd.concat([cloudy_df,clear_df,rain_df,snow_df],axis=0)

weather_df.drop(columns=['Date/Time','Weather'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
weather_df['Std_Weather'] = label_encoder.fit_transform(weather_df['Std_Weather'])

X = weather_df.drop(['Std_Weather'],axis=1)
Y = weather_df['Std_Weather']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(X)

scale = StandardScaler()
scale.fit(X)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_std,Y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model_random = RandomForestClassifier(n_estimators=200,min_samples_split=2,min_samples_leaf=1,max_depth=20,bootstrap=True)
model_random.fit(x_train,y_train)

y_pred = model_random.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

with open('model.pkl', 'wb') as f:
  pickle.dump(model_random, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scale, f)


