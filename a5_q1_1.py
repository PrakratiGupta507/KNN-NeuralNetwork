# -*- coding: utf-8 -*-
"""A5_Q1_1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11bNAtDFED49Y2L4pX0n0HOyF3hlPGb78
"""

from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#read the dataset and convert in csv
data = pd.read_csv('/content/gdrive/My Drive/ML_Assignment5/sat.trn',header=None,delimiter=" ")
test = pd.read_csv('/content/gdrive/My Drive/ML_Assignment5/sat.tst',header=None,delimiter=" ")

X=data.iloc[:,:-1]
y=data[36]

tsne_results = TSNE(n_components =2).fit_transform(X)

tsne_df = pd.DataFrame({'X':tsne_results[:,0],
                        'Y':tsne_results[:,1],
                        'target':y})
tsne_df.head()

fig=plt.figure(figsize=(14,8))
import seaborn as sns
sns.scatterplot(x="X", y="Y",
              hue="target",
              palette=['purple','red','orange','brown','blue',
                       'dodgerblue'],
              legend='full',
              data=tsne_df)

