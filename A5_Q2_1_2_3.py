import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import h5py
f = h5py.File('MNIST_Subset.h5','r')
f.keys()
images = f['X'][:]
labels = f['Y'][:]
images.shape,labels.shape
main_lables = np.unique(labels,axis=0)
print(main_lables)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = np.reshape(X_train,(11400,784))
X_test = np.reshape(X_test,(2851,784))
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 50),activation='logistic')
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
Accuracy = accuracy_score(y_test, predictions)
Loss = mean_squared_error(y_test, predictions)
print("Accuracy is : ",Accuracy)
print("Loss is : ",Loss)
model = TSNE(n_components = 2).fit_transform(X_train)
joblib.dump(model,'TSNEmodelQ2(3).pkl')
model1 = joblib.load('TSNEmodelQ2(3).pkl')
data=model1 
data = pd.DataFrame(data)
x=data[0]
y=data[1]
z=y_train
df = pd.DataFrame({"Zero_value":x,"One_value":y,"labels":z})
df.to_csv('TSNEdf.csv')
file1 = pd.read_csv("TSNEdf.csv")
plt.figure(figsize=(15,10))
sns.scatterplot(data=file1,x="Zero_value", y="One_value",hue='labels',palette="bright")
plt.xlabel("sampel-0")
plt.ylabel("sampel-1")
plt.legend()
plt.show()
alphas = [0.0001,0.1,5]
Y_train = df['labels']
x_train = df.drop('labels',axis=1)
classifier = []
for i in alphas:
    clf = MLPClassifier(hidden_layer_sizes=(100, 50, 50),activation='logistic',alpha=i).fit(x_train,Y_train)
    classifier.append(clf)
classifier = np.array(classifier)
for i in range(classifier.shape[0]):
    minimum1, maximum1 = X[:, 0].min()-1, X[:, 0].max()+1
    minimum2, maximum2 = X[:, 1].min()-1, X[:, 1].max()+1
    x1_gd = np.arange(minimum1, maximum1, 0.1)
    x2_gd = np.arange(minimum2, maximum2, 0.1)
    x_x, y_y = np.meshgrid(x1_gd, x2_gd)
    r_1, r_2 = x_x.flatten(), y_y.flatten()
    r_1, r_2 = r_1.reshape((len(r_1), 1)), r_2.reshape((len(r_2), 1))
    grid_train = np.hstack((r_1,r_2))
    y_pred = classifier[i].predict(grid_train)
    z_z = y_pred.reshape(x_x.shape)
    plt.figure(figsize=(15,10))
    plt.contourf(x_x, y_y, z_z, cmap='Pastel1')
    sns.scatterplot(x=X[:,0],y=X[:,1],c=y,cmap='Paired',hue=y)
    plt.legend()
    titles = ['Decision Boundary Alpha = 0.0001','Decision Boundary Alpha = 0.1','Decision Boundary Alpha = 5']
    plt.xlabel('sample - 1',fontweight='bold',fontsize=14.0)
    plt.ylabel('sample - 2',fontweight='bold',fontsize=14.0)
    plt.title(titles[i],fontweight='bold',fontsize=14.0)
    plt.show()




