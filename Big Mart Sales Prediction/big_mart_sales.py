import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

sales_data = pd.read_csv("Train.csv")
#filling the missing values in 'Item Weight column with 'mean' value
sales_data['Item_Weight'].fillna(sales_data['Item_Weight'].mean(), inplace =True)

sales_data['Item_Identifier'] = sales_data['Item_Identifier'].str.slice(0,2)

sales_data['Outlet_Size'].fillna('Medium',inplace=True)

sales_data['Outlet_Identifier'] = sales_data['Outlet_Identifier'].str.slice(-2,)

sales_data = sales_data.replace({'Item_Identifier':{'FD':1,'NC':2,'DR':3},'Item_Type':{'Others':0,'Fruits and Vegetables':1,'Snack Foods':2,'Household':3,'Frozen Foods':4,'Dairy':5,'Canned':6,'Baking Goods':7,'Health and Hygiene':8,'Soft Drinks':9,'Meat':10,'Breads':11,'Hard Drinks':12,'Starchy Foods':13,'Breakfast':14,'Seafood':15},'Outlet_Type':{'Supermarket Type1':1,'Grocery Store':2,'Supermarket Type3':3,'Supermarket Type2':4},'Outlet_Location_Type':{'Tier 1':1,'Tier 2':2,'Tier 3':3},'Item_Fat_Content':{'Low Fat':1,'Regular':2,'LF':3,'reg':4,'low fat':5},'Outlet_Size':{'Small':0,'Medium':1,'High':2}})

X = sales_data.drop(columns=['Outlet_Establishment_Year','Outlet_Identifier','Item_Outlet_Sales'],axis=1)
Y = sales_data['Item_Outlet_Sales']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

model =XGBRegressor(enable_categorical=True)
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
X_train_error = metrics.r2_score(Y_train,X_train_prediction)
print("The error in the training dataset is :",X_train_error)

X_test_prediction = model.predict(X_test)
X_test_error = metrics.r2_score(Y_test,X_test_prediction)
print("The error in the test dataset is :",X_test_error)

input_data =['FDH17',16.2,'Regular',0.016687114,'Frozen Foods',96.9726,'Medium','Tier 2','Supermarket Type1']
input_data[0] = input_data[0][0:2]

#sales_data = sales_data.replace({'Item_Identifier':{'FD':1,'NC':2,'DR':3},'Item_Type':{'Others':0,'Fruits and Vegetables':1,'Snack Foods':2,'Household':3,'Frozen Foods':4,'Dairy':5,'Canned':6,'Baking Goods':7,'Health and Hygiene':8,'Soft Drinks':9,'Meat':10,'Breads':11,'Hard Drinks':12,'Starchy Foods':13,'Breakfast':14,'Seafood':15},'Outlet_Type':{'Supermarket Type1':1,'Grocery Store':2,'Supermarket Type3':3,'Supermarket Type2':4},'Outlet_Location_Type':{'Tier 1':1,'Tier 2':2,'Tier 3':3},'Item_Fat_Content':{'Low Fat':1,'Regular':2,'LF':3,'reg':4,'low fat':5},'Outlet_Size':{'Small':0,'Medium':1,'High':2}})

#'Item_Identifier':{'FD':1,'NC':2,'DR':3}
if(input_data[0] == 'FD'):
    input_data[0] = 1
elif(input_data[0] == 'NC'):
    input_data[0] = 2
elif(input_data[0] == 'DR'):
    input_data[0] = 3


#{'Low Fat':1,'Regular':2,'LF':3,'reg':4,'low fat':5}
if(input_data[2] == 'Low Fat'):
    input_data[2] = 1
elif(input_data[2] == 'Regular'):
    input_data[2] = 2
elif(input_data[2] == 'Low Fat'):
    input_data[2] = 3
elif(input_data[2] == 'reg'):
    input_data[2] = 4
elif(input_data[2] == 'low fat'):
    input_data[2] = 5

#Item_Type':{'Others':0,'Fruits and Vegetables':1,'Snack Foods':2,'Household':3,'Frozen Foods':4,'Dairy':5,'Canned':6,'Baking Goods':7,'Health and Hygiene':8,'Soft Drinks':9,'Meat':10,'Breads':11,'Hard Drinks':12,'Starchy Foods':13,'Breakfast':14,'Seafood':15}
if(input_data[4] == 'Others'):
    input_data[4] = 0
elif(input_data[4] == 'Fruits and Vegetables'):
    input_data[4] = 1
elif(input_data[4] == 'Snack Foods'):
    input_data[4] = 2
elif(input_data[4] == 'Household'):
    input_data[4] = 3
elif(input_data[4] == 'Frozen Foods'):
    input_data[4] = 4
elif(input_data[4] == 'Dairy'):
    input_data[4] = 5
elif(input_data[4] == 'Canned'):
    input_data[4] = 6
elif(input_data[4] == 'Baking Goods'):
    input_data[4] = 7
elif(input_data[4] == 'Health and Hygiene'):
    input_data[4] = 8
elif(input_data[4] == 'Soft Drinks'):
    input_data[4] = 9
elif(input_data[4] == 'Meat'):
    input_data[4] = 10
elif(input_data[4] == 'Breads'):
    input_data[4] = 11
elif(input_data[4] == 'Hard Drinks'):
    input_data[4] = 12
elif(input_data[4] == 'Starchy Foods'):
    input_data[4] = 13
elif(input_data[4] == 'Breakfast'):
    input_data[4] = 14
elif(input_data[4] == 'Seafood'):
    input_data[4] = 15

#Supermarket Type1':1,'Grocery Store':2,'Supermarket Type3':3,'Supermarket Type2':4
if(input_data[8] == 'Supermarket Type1'):
    input_data[8] = 1
elif(input_data[8] == 'Grocery Store'):
    input_data[8] = 2
elif(input_data[8] == 'Supermarket Type3'):
    input_data[8] = 3
elif(input_data[8] == 'Supermarket Type2'):
    input_data[8] = 4

#Outlet_Location_Type':{'Tier 1':1,'Tier 2':2,'Tier 3':3}
if(input_data[7] == 'Tier 1'):
    input_data[7] = 1
elif(input_data[7] == 'Tier 2'):
    input_data[7] = 2
elif(input_data[7] == 'Tier 3'):
    input_data[7] = 3

#Outlet_Size':{'Small':0,'Medium':1,'High':2
if(input_data[6] == 'Small'):
    input_data[6] = 0
elif(input_data[6] == 'Medium'):
    input_data[6] = 1
elif(input_data[6] == 'High'):
    input_data[6] = 2

input_data_np = np.asanyarray(input_data)
input_reshaped = input_data_np.reshape(1,-1)

prediction =model.predict(input_reshaped)
print(" The predicted sales for the following is :",prediction)

plt.scatter(X_test_prediction,Y_test)
plt.xlabel("Predicted value")
plt.ylabel("The true value")
plt.show()