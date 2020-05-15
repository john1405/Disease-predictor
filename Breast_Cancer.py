    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn import model_selection
    from sklearn.metrics import classification_report, accuracy_score
    from pandas.plotting import scatter_matrix
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler


    a=pd.read_csv("Breast_cancer_data.csv")     #load Dataset

    a.head()

    print(a.shape)
    
    a.hist(figsize = (8,8))
    plt.show()

    scatter_matrix(a, figsize = (10,10))
    plt.show()

    X = a.iloc[:,:-1].values
    y = a.iloc[:,5].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    clsr = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
    clsr.fit(X_train, y_train)
    
    clf=SVC()
    clf.fit(X_train,y_train)
    
    models=[]
    models.append(("KNN",clsr))
    models.append(("SVM",clf))
    
    for name, model in models:
        y_pred = model.predict(X_test)
        print(name)
        print(accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    cm = metrics.confusion_matrix(y_test, y_pred) 
    print(cm)
    accuracy = metrics.accuracy_score(y_test, y_pred) 
    print("Accuracy score:",accuracy)
    precision = metrics.precision_score(y_test, y_pred)
    print("Precision score:",precision)
    recall = metrics.recall_score(y_test, y_pred) 
    print("Recall score:",recall)

    ###EXAMPLE TEST CASE###
    print("Example Test Case")
    arr=np.array([[17.95,20.01,114.2,982,0.08402]])
    arr=arr.reshape(len(arr), -1)
    prediction = clf.predict(arr)
    if prediction==0:
        print('Malignant')
    elif prediction==1:
        print('Benign')