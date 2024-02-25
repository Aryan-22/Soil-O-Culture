import warnings 
import numpy as np
import pandas as pd
from sklearn import metrics
warnings.filterwarnings("ignore")
df = pd.read_csv("crop.csv")
features = df[["NITROGEN","PHOSPHORUS","POTASSIUM","TEMPERATURE","HUMIDITY","PH","RAINFALL"]]
labels = df["CROP"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(features,labels,test_size = 0.25)
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train,Y_train)



rf = pd.read_csv("aslirainfall.csv")
'''print(rf)
from pyowm import OWM
from datetime import date
todays_date = date.today()

# printing todays date
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
actualm = months[todays_date.month % 12 - 1].upper()
m = todays_date.month % 12
print("Current month:", actualm)




state = input("enter your state:").upper()
city = input("enter your city:").capitalize()
owm = OWM('c706f5b72e2e8e0eaf78a1a673fb53ee') #using the api key
mgr = owm.weather_manager()
weatheratplace = mgr.weather_at_place(city)
weather = weatheratplace.weather
temp = weather.temperature("celsius") 
humidity = weather.humidity
actualtemp = temp["temp"]
d = rf["STATES"].tolist()
i = d.index(state)
rainfall = float(rf.loc[i,actualm])
    
print("temperature = {0} , humidity = {1} ,rainfall (mm) = {2}".format(actualtemp,humidity,rainfall))


print("enter N,P,K,pH")
l = list(map(float,input().split()))
N,P,K,pH = l
x = [N,P,K,actualtemp,humidity,pH,rainfall]

a = np.array([x])
unique_items = list(dict.fromkeys(labels)) #stores all the crops (unique)
prediction = RF.predict(a)

print("best crop to grow --> {}".format(prediction[0]))'''
unique_items = list(dict.fromkeys(labels))
print(unique_items)
