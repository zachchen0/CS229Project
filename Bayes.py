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


def bayes():
    with open('elite_data_25000.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = df[['elite', 'text']]

    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(df.text)
    X_train, X_test, y_train, y_test = train_test_split(X_train_counts, df.elite, test_size=0.1666, random_state=22)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)

    # For Finding Top 10 Features
    # X_train_counts = tfidf_transformer.fit_transform(X_train_counts)
    # features = count_vect.get_feature_names_out()
    # X_train_counts = X_train_counts[0].toarray().flatten()
    # top10 = np.argsort(X_train_counts)[::-1][:10].flatten()
    # topfeats = [(features[i], X_train_counts[i]) for i in top10]
    # topfeatsframe = pd.DataFrame(topfeats)
    # print(topfeatsframe)

    # clf = MultinomialNB().fit(X_train_tfidf, y_train)
    # X_new_tfidf = tfidf_transformer.transform(X_test)
    # predicted = clf.predict(X_new_tfidf)
    #
    #
    # print('Naive-Bayes (tf-idf)')
    # print('Accuracy: ', np.mean(predicted == y_test))
    # print('Recall: ', recall_score(y_test, clf.predict(X_test)))
    # print('Precision: ', precision_score(y_test, clf.predict(X_test)))
    # print('F1: ', f1_score(y_test, clf.predict(X_test)))
    # print()

    text_clf = MultinomialNB()
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)

    print('Naive-Bayes (word-counts)')
    print('Accuracy: ', np.mean(predicted == y_test))
    print('Recall: ', recall_score(y_test, text_clf.predict(X_test)))
    print('Precision: ', precision_score(y_test, text_clf.predict(X_test)))
    print('F1: ', f1_score(y_test, text_clf.predict(X_test)))

    nb_probs = text_clf.predict_proba(X_test)
    nb_probs = nb_probs[:, 1]
    nb_auc = roc_auc_score(y_test, nb_probs)
    print('AUROC: ', nb_auc)

    nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes, AUROC = %0.3f' % nb_auc)
    plt.title('ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


    # Elite
    # Naive-Bayes (tf-idf)
    # Accuracy:  0.6475390156062425
    # Recall:  0.7980815347721822
    # Precision:  0.6255639097744361
    # F1:  0.7013698630136987

    # Naive-Bayes (word-counts)
    # Accuracy:  0.6506602641056423
    # Recall:  0.7453237410071942
    # Precision:  0.6271186440677966
    # F1:  0.6811308349769887

    # 60000 Examples
    # Naive-Bayes (tf-idf)
    # Accuracy:  0.6191476590636255
    # Recall:  0.16771844660194174
    # Precision:  0.6120460584588131
    # F1:  0.2632882453800724
    #
    # Naive-Bayes (word-counts)
    # Accuracy:  0.6226490596238495
    # Recall:  0.4529126213592233
    # Precision:  0.5514184397163121
    # F1:  0.4973347547974414

    # 600000 Examples
    # Naive-Bayes (tf-idf)
    # Accuracy:  0.6239195678271309
    # Recall:  0.30950824265633026
    # Precision:  0.6063586459286368
    # F1:  0.40982592833070525
    #
    # Naive-Bayes (word-counts)
    # Accuracy:  0.6259003601440576
    # Recall:  0.4676831831130622
    # Precision:  0.5784214630201866
    # F1:  0.5171910707138522

bayes()