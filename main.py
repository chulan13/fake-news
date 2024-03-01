#importing libraries
import numpy as np, pandas as pd, matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

#importing database
dataset = pd.read_csv("data_set_4.csv")
print(dataset.size)

'''
# dataset = dataset.dropna(subset=['Label'])
# print(dataset.size)
'''
#drop all null values
nan_rows = dataset[dataset.isna().any(axis=1)]

#convert boolean values into binary for Fake News False --> 0, for True News True --> 1
dataset['Label'] = dataset['Label'].astype(int)

#count the amount of unique values
label_counts = dataset['Label'].value_counts()

'''
dataset.where(filterT, inplace=True)
dataset.where(filterF, inplace=True)
dataset[['StringPart', 'CharPart']] = dataset['Link'].str.extract(r'([a-zA-Z]+) ([a-zA-Z]+)')
'''
print(label_counts)

'''
# Plot a bar chart
plt.bar(label_counts.index, label_counts.values, color=['red', 'green'])
# Add labels and title
plt.xlabel('News Type')
plt.ylabel('Number of News Articles')
plt.xticks([0, 1], ['Fake', 'True'])
plt.title('Distribution of Fake and True News')
plt.show()
'''

#splitting into train and test
true_false = dataset.Label
x_train,x_test,y_train,y_test=train_test_split(dataset['Text'], true_false, test_size=0.2, random_state=7)
print('x train=',len(x_train),
      "\t",'x test=',len(x_test),
      "\t",'y train=',len(y_train),
      "\t",'y test=',len(y_test))


file_path = 'ukrainian'
# Check if the file exists
if os.path.isfile(file_path):
    with open(file_path, "r") as f:
        raw_ukr = f.read()
        # print("Content of 'ukrainian' file:\n", vocabulary)
else:
    print("File 'ukrainian' does not exist.")  

stopwords = raw_ukr.split("\n")
print(stopwords)

# TfidfVectorizer
# PassiveAggressiveClassifier
