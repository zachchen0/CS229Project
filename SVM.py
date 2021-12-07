import numpy as np
import pandas as pd
import json

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

with open('elite_data_25000.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[['elite', 'text']]

X_train, X_test, y_train, y_test = train_test_split(df.text, df.elite, test_size=0.1666, random_state=22)

text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                           alpha=1e-3, random_state=22,
                           max_iter=5, tol=None)),
 ])

text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

print('SVM')
print('Accuracy: ', np.mean(predicted == y_test))
print('Recall: ', recall_score(y_test, text_clf.predict(X_test)))
print('Precision: ', precision_score(y_test, text_clf.predict(X_test)))
print('F1: ', f1_score(y_test, text_clf.predict(X_test)))
print()


# Graphing
calibrator = CalibratedClassifierCV(text_clf, cv='prefit')
model = calibrator.fit(X_train, y_train)
nb_probs = model.predict_proba(X_test)
nb_probs = nb_probs[:, 1]
nb_auc = roc_auc_score(y_test, nb_probs)
print('AUROC: ', nb_auc)

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

plt.plot(nb_fpr, nb_tpr, label='SVM, AUROC = %0.3f' % nb_auc)
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Elite
# SVM
# Accuracy:  0.6871548619447779
# Recall:  0.6714628297362111
# Precision:  0.6937561942517344
# F1:  0.6824274920789666



