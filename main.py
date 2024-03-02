#importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np, pandas as pd, matplotlib as mlt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

#importing database
dataset = pd.read_csv("fake-news/data_set_4.csv")
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


file_path = 'fake-news/ukrainian'
# Check if the file exists
if os.path.isfile(file_path):
    with open(file_path, "r") as f:
        raw_ukr = f.read()
        # print("Content of 'ukrainian' file:\n", vocabulary)
else:
    print("File 'ukrainian' does not exist.")  

stopwords = raw_ukr.split("\n")
print(stopwords)

# TfidfVectorizer - Term Frequency Inverse Document Frequency - an overall document weightage:
#                   TF-IDF(t) = TF(t,d) * IDF(t)
# CountVectorizer - gives number of frequency with respect to index of vocabulary.
# TfidfTransformer
'''
ContVectorizer() gives number of frequency with respect to index of vocabulary. 
Tf-idf conciders overall documents of weight of words.

TF-IDF = TF(t,d) * IDF(t),      where TF(t,d) is a number of times term 't' appears in a doc 'd'; 
                                and IDF(t) = log((1+n) / (1+df(d,t)) +1), where n = amount of documents, 
                                df(d,t) - document frequency of the term t

                                Term-Frequency Func:        TF(t,d) = sum(fr(x,t)) in range x є d   
                                                            where fr(x,t)={[1, if x=t] or [0, otherwise]
                                
                                                            TF(t,d) returns how many times the term 't' is present in
                                                            document 'd' 

        The TF-IDF weight:
                The idf(inverse document frequency) is defined: IDF(t) = log(|D| / 1+|{d : t є d}|),
                where |{d : t є d}| is the number of documents where the term 't' appears, when TF(t,d) != 0
'''

tfidfVectorizer = TfidfVectorizer(stop_words=stopwords,max_df=0.7,strip_accents=None)
tfidf_train=tfidfVectorizer.fit_transform(x_train.values.astype('U')) 
tfidf_test=tfidfVectorizer.transform(x_test.values.astype('U'))

# PassiveAggressiveClassifier
