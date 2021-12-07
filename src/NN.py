#################################################################################################
#################################################################################################
#################################################################################################
# RUN THIS CODE ON GOOGLE COLAB
# https://colab.research.google.com/drive/1sJEb6r5lRpUaa8Xm8TDETRvsQG4zRqlT?usp=sharing
#################################################################################################
#################################################################################################
#################################################################################################


import pandas as pd
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split

with open('elite_data_25000.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df[['elite', 'text']]

X_train, X_test, y_train, y_test = train_test_split(df.text, df.elite, test_size=0.1666, random_state=22)

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

dropout = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
layer1 = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(dropout)

model = tf.keras.Model(inputs=[text_input], outputs=[layer1])
model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
    tf.keras.metrics.Precision(name='Precision'),
    tf.keras.metrics.Recall(name='Recall')
]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)
model.fit(X_train, y_train, epochs=10)

model.evaluate(X_test, y_test)
