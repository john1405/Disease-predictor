import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import regularizers as reg
%matplotlib inline
from sklearn.metrics import classification_report, accuracy_score

url = "heart.csv"
cleveland = pd.read_csv(url)
print('Shape of DataFrame: {}'.format(cleveland.shape))

a=cleveland[~cleveland.isin(['?'])]
a.head()

a=a.dropna(axis=0)
print(a.loc[1000:])

print(a.shape)
print(a.dtypes)

a=a.apply(pd.to_numeric)
print(a.dtypes)

a.describe()

a.hist(figsize=(10,10))
plt.show()

pd.crosstab(a.age,a.target).plot(kind="bar",figsize=(25,7))
plt.title('Heart Disease Distribution With Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(a.corr(),annot=True,fmt='0.1f')
plt.show()

age_unique=sorted(a.age.unique())
age_thalach_values=a.groupby('age')['thalach'].count().values
mean_thalach=[]
for i,age in enumerate(age_unique):
    mean_thalach.append(sum(a[a['age']==age].thalach)/age_thalach_values[i])
    
plt.figure(figsize=(12,6))
sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)
plt.xlabel('Age',fontsize=15,color='blue')
plt.xticks(rotation=45)
plt.ylabel('Thalach',fontsize=15,color='blue')
plt.title('Age V/S Thalach',fontsize=20,color='blue')
plt.grid()
plt.show()

X=np.array(a.drop(['target'],1))
Y=np.array(a['target'])
X[0]

mean=X.mean(axis=0)
X-=mean
std=X.std(axis=0)
X/=std
X[:2]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, stratify=Y, random_state=42, test_size = 0.2)
Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()

Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

y_train=to_categorical(y_train,num_classes=None)
y_test=to_categorical(y_test,num_classes=None)
print(y_train.shape)
print(y_train[:10])

X_train[0]

def create_model():
    model=Sequential()
    model.add(Dense(16,input_dim=13,kernel_initializer='normal',kernel_regularizer=reg.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2,activation='softmax'))
    adam=Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
model = create_model()

print(model.summary())

history=model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=50, batch_size=10)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

print(Y_train_binary[:20])

def create_binary_model():
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=reg.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=reg.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())

history=binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=50, batch_size=10)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

categorical_pred = to_categorical(np.argmax(model.predict(X_test), axis=1))

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

from sklearn.metrics import classification_report, accuracy_score
binary_pred = np.round(binary_model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))


###EXampLE TesT CasE###
arr=np.array([[52,1,2,138,223,0,1,169,0,0,2,4,2]])
arr=arr.reshape(len(arr), -1)
prediction=model.predict(arr)
if prediction is 0:
    print("No Heart Problem")
else :
    print("Heart Problem, Need to Consult Doctor")