from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

def print_metrics(y, y_predict):
    print(confusion_matrix(y, y_predict))
    metrics = precision_recall_fscore_support(y, y_predict, average='micro')
    print('precision: %.3f recall: %.3f f-score %.3f' %(metrics[0], metrics[1], metrics[2]))

# Computes for k = [3, 5, 7]
def knn_class(X, Y, x, y):
    for k in [3, 5, 7]:
        print('training for k = ' + str(k) + '...')
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X, Y)
        y_predict = model.predict(x)

        print('confusion matrix k = ' + str(k) + ':')
        print_metrics(y, y_predict)

# Computes for alpha = [0.5...1.5]
def naives_class(X, Y, x, y):
    for alpha in [0.5, 1, 1.5]:
        print('training for alpha = ' + str(alpha) + '...')
        model = MultinomialNB(alpha = alpha)
        model.fit(X, Y)
        y_predict = model.predict(x)

        print('confusion matrix alpha = ' + str(alpha) + '...')
        print_metrics(y, y_predict)

# Computes for penalty = ['l2', 'none', 'l1']
def logistic_class(X, Y, x, y):
    for penalty in ['l2', 'none', 'l1']:
        print('training with penalty ' + penalty + '...')
        if penalty == 'l1':
            model = LogisticRegression(penalty = penalty, solver = 'liblinear')
        else:
            model = LogisticRegression(penalty = penalty)
        model.fit(X, Y)
        y_predict = model.predict(x)

        print('confusion matrix penalty = ' + penalty + ':')
        print_metrics(y, y_predict)

path = 'C:/Users/hp/Desktop/codigo/'
file = 'data_wo.csv'
data = pd.read_csv(path + file, header = 0)
tags = data['Sentiment']
data = data[data.columns[1:]]

print(' creating samples...')
X, x, Y, y = train_test_split(data, tags, test_size = 0.2)

print(' training with k-nearest neighbors...')
knn_class(X, Y, x, y)

print(' training with logistic regression...')
logistic_class(X, Y, x, y)

print(' training with multimodal naive bayes...')
naives_class(X, Y, x, y)
