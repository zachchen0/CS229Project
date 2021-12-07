import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

with open('elite_data_25000.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[['elite', 'text']]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df.text)
X_train, X_test, y_train, y_test = train_test_split(X_train_counts, df.elite, test_size=0.1666, random_state=22)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)

for c in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
    model = LogisticRegression(solver='liblinear', penalty='l2', C=c)
    model.fit(X_train_tfidf, y_train)
    predicted = model.predict(X_test_tfidf)

    print('Logistic Regression, ', 'c = ', c)
    print('Accuracy: ', np.mean(predicted == y_test))
    print('Recall: ', recall_score(y_test, predicted))
    print('Precision: ', precision_score(y_test, predicted))
    print('F1: ', f1_score(y_test, predicted))
    print()



model = LogisticRegression(solver='liblinear', penalty='l2', C=1)
model.fit(X_train_tfidf, y_train)
nb_probs = model.predict_proba(X_test)
nb_probs = nb_probs[:, 1]
nb_auc = roc_auc_score(y_test, nb_probs)
print('AUROC: ', nb_auc)

nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

plt.plot(nb_fpr, nb_tpr, label='Logistic Regression, AUROC = %0.3f' % nb_auc)
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# C = 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100
# Elite
# Logistic Regression,  c =  0.01
# Accuracy:  0.6686674669867947
# Recall:  0.6709832134292566
# Precision:  0.6684185379837554
# F1:  0.6696984202967928
#
# Logistic Regression,  c =  0.03
# Accuracy:  0.6813925570228091
# Recall:  0.6729016786570743
# Precision:  0.68505859375
# F1:  0.6789257198161142
#
# Logistic Regression,  c =  0.1
# Accuracy:  0.6912364945978391
# Recall:  0.6853717026378897
# Precision:  0.694026226323458
# F1:  0.6896718146718146
#
# Logistic Regression,  c =  0.3
# Accuracy:  0.697719087635054
# Recall:  0.6959232613908873
# Precision:  0.6989402697495183
# F1:  0.6974285027637588
#
# Logistic Regression,  c =  1
# Accuracy:  0.7034813925570228
# Recall:  0.7002398081534772
# Precision:  0.7053140096618358
# F1:  0.7027677496991576
#
# Logistic Regression,  c =  3
# Accuracy:  0.690516206482593
# Recall:  0.6848920863309352
# Precision:  0.6932038834951456
# F1:  0.6890229191797346
#
# Logistic Regression,  c =  10
# Accuracy:  0.6753901560624249
# Recall:  0.6647482014388489
# Precision:  0.6797449730259931
# F1:  0.6721629485935985
#
# Logistic Regression,  c =  30
# Accuracy:  0.6542617046818727
# Recall:  0.6345323741007194
# Precision:  0.6611694152923538
# F1:  0.6475770925110131
#
# Logistic Regression,  c =  100
# Accuracy:  0.6391356542617047
# Recall:  0.6206235011990408
# Precision:  0.6450648055832503
# F1:  0.6326081642630164











# 600000 Examples C=100
# Logistic Regression w/o Regularization
# Accuracy:  0.6279311724689876
# Recall:  0.4732639051043758
# Precision:  0.5807117070654977
# F1:  0.52151090983944

# 600000 Examples C=30
# Logistic Regression
# Accuracy:  0.6347238895558224
# Recall:  0.4764395460701443
# Precision:  0.5915060153645456
# F1:  0.5277738259981117

# 600000 Examples C=10
# Logistic Regression
# Accuracy:  0.6419567827130852
# Recall:  0.47737355811889975
# Precision:  0.6039229587616685
# F1:  0.5332429119173688

# 600000 Examples C=3
# Logistic Regression
# Accuracy:  0.6489095638255302
# Recall:  0.4772801569140242
# Precision:  0.6166098524842378
# F1:  0.5380717341230669

# 600000 Examples C=1
# Logistic Regression
# Accuracy:  0.6513105242096838
# Recall:  0.4725166954653715
# Precision:  0.6226269960924279
# F1:  0.5372841079559786

# 600000 Examples C=0.3
# Logistic Regression
# Accuracy:  0.6513805522208883
# Recall:  0.4644141409424182
# Precision:  0.6254402515723271
# F1:  0.5330313831640447

# 600000 Examples C=0.1
# Logistic Regression
# Accuracy:  0.649359743897559
# Recall:  0.45278569093541304
# Precision:  0.6253950848222924
# F1:  0.5252735941055369

# 600000 Examples C=0.03
# Logistic Regression
# Accuracy:  0.6452881152460984
# Recall:  0.4334516415261757
# Precision:  0.6238196054709816
# F1:  0.5114971825358556

# 600000 Examples C=0.01
# Logistic Regression
# Accuracy:  0.6390656262505002
# Recall:  0.4020221360855555
# Precision:  0.6218441868024704
# F1:  0.4883354842369491







