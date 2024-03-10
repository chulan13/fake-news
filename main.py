#importing libraries
from flask import Flask, render_template, request
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__, template_folder='templates')


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
dataset['Text'] = dataset['Text'].str.lower()
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
# make the text lowercase, drop all special characters exception for apostroph


#splitting into train and test
true_false = dataset.Label
x_train,x_test,y_train,y_test=train_test_split(dataset['Text'], true_false, test_size=0.2, random_state=7)


# print('x train=',len(x_train),
#       "\t",'x test=',len(x_test),
#       "\t",'y train=',len(y_train),
#       "\t",'y test=',len(y_test))


file_path = 'fake-news/ukrainian'
# Check if the file exists
if os.path.isfile(file_path):
    with open(file_path, "r") as f:
        raw_ukr = f.read()
        # print("Content of 'ukrainian' file:\n", vocabulary)
else:
    print("File 'ukrainian' does not exist.")  

stopwords = raw_ukr.split("\n")
# print(stopwords)

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

tfidfVectorizer = TfidfVectorizer(stop_words=stopwords,max_df=0.8,strip_accents=None)
tfidf_train=tfidfVectorizer.fit_transform(x_train.values.astype('U')) 
tfidf_test=tfidfVectorizer.transform(x_test.values.astype('U'))


# PassiveAggressiveClassifier 
'''
Passive Aggressive Algorithms:
    It's one of the 'online-learning algorithms', 
    so the input data comes into sequential order and the machine learning model is updated step by step.
    They're somewhat similar to a Perception model (do not require a learning rate).

    Passive when prediction is correct (doesnt change anything)
    Aggressive when prediction is incorrect (makes changes)

            Important parameters:
                    * C: the regularization parameter, 
                    and denotes the penelization the model will make on an incorrect prediction.
                    * max_iter
                    * tol: the stopping criterion. 
                    When set to None will stop when loss > prev_loss - tol (default 1e-3)

    Math:
        Training data (X,y) where 'X' represents TFIFD matrix and 'y' repr. corresponding target labels.
        Weight vector 'w' - it's what Passive Aggressive alg tries to find.

        The hinge loss: L(w) = max(0, 1 - y * (w^t * x)), 
        where w^T - the traspose of the weight vector w, x is a feature vector, and y is the corresp. true label (-1 or 1).

        The update rule when missclassification occures:
                    w_new = w_old + (learning_rate * hinge_loss * x), 
                    where learning_rate is a hyperparameter controlling the step size of the update,
                    and x is a feature vector of the missclassified instance.

        An upper bound or margin parameter C - to ensure the model doesnt overfit.
                    w_new = w_old + (min(loss,C) * learning_rate * x)
'''
paclass=PassiveAggressiveClassifier(max_iter=50)
paclass.fit(tfidf_train,y_train)

# Predictions
y_predPAC=paclass.predict(tfidf_test)
pac_accuracy=accuracy_score(y_test,y_predPAC)
print(f'Passive Aggressive Accuracy: {round(pac_accuracy*100,2)}%')

misclassified_indices = np.where(y_test != y_predPAC)[0]
misclassified_x = x_test.iloc[misclassified_indices]
misclassified_y = y_test.iloc[misclassified_indices]

# Add misclassified examples to training data
x_train = pd.concat([x_train, misclassified_x])
y_train = pd.concat([y_train, misclassified_y])

# Retrain the model with the updated training data
tfidf_train = tfidfVectorizer.transform(x_train.values.astype('U'))
paclass.fit(tfidf_train, y_train)

# Predictions after retraining
y_pred_after_retrain = paclass.predict(tfidf_test)

# Evaluate accuracy after retraining
accuracy_after_retrain = accuracy_score(y_test, y_pred_after_retrain)
print(f'Accuracy after Retraining: {round(accuracy_after_retrain * 100, 2)}%')

# lets try this shit
'''
trying = input()
trying = [trying]
tfidf_try = tfidfVectorizer.transform(np.asarray(trying, dtype='U'))
y_tryPAC=paclass.predict(tfidf_try)
print(y_tryPAC)
'''


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']

    # Your existing code for processing user_input and making predictions


    # Check if is not empty
    if user_input.strip():  # if contains non-whitespace characters
        user_text = [user_input]

        # Convert into Unicode
        tfidf_try = tfidfVectorizer.transform(np.asarray(user_text, dtype='U'))
        y_tryPAC = paclass.predict(tfidf_try)

        print(f'Predicted Label: {y_tryPAC}')
    else:
        print("lol wtf ziom")


    return render_template('result.html', prediction=y_tryPAC)

if __name__ == '__main__':
    app.run(debug=True)


# Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(tfidf_train, y_train)

# Predictions
y_predRF = random_forest_classifier.predict(tfidf_test)
rf_accuracy = accuracy_score(y_test, y_predRF)
print(f'Random Forest Accuracy: {round(rf_accuracy * 100, 2)}%')



# SVM
svm_classifier = LinearSVC()
svm_classifier.fit(tfidf_train, y_train)

# Predictions
y_predSVM = svm_classifier.predict(tfidf_test)
svm_accuracy = accuracy_score(y_test, y_predSVM)
print(f'SVM Accuracy: {round(svm_accuracy * 100, 2)}%')



# Logistic Regression
logreg_model = LogisticRegression(max_iter=100)
logreg_model.fit(tfidf_train, y_train)

# Predictions
y_predLR = logreg_model.predict(tfidf_test)
logreg_accuracy = accuracy_score(y_test, y_predLR)
print(f'Logistic Regression Accuracy: {round(logreg_accuracy * 100, 2)}%')

predict_list = {

}


dict_live = { 
    0 : 'fake',
    1 : 'true'
}
'''
cm = confusion_matrix(y_test, y_predPAC)
df_cm = pd.DataFrame(cm, index = [dict_live[i] for i in range(0,2)], columns = [dict_live[i] for i in range(0,2)])
plt.figure(figsize = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted Class", fontsize=18)
plt.ylabel("True Class", fontsize=18)
plt.title('Passive Aggressive Classifier', fontsize = 15)
plt.show()
'''

