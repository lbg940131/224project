import pandas as pd
import numpy as np

import os
import warnings

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("_NA_")

X = TfidfVectorizer(stop_words='english', max_features=50000).fit_transform(df)

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    

def get_AUC(model,data,y_gt):
	y_pred = model.predict_proba(data)[:,1]
	score = roc_auc_score(y_gt, y_pred)
	return score

preds = np.zeros((test.shape[0], 6))
num_train = train.shape[0]

L=[]

start= time.time()
for idx, name in enumerate(classes):
    print('===Fit '+name)
    model = LogisticRegression()
    model.fit(X[:num_train], train[name])
    preds[:,idx] = model.predict_proba(X[num_train:])[:,1]
    score = get_AUC(model,X[:num_train], train[name])
    print('AUC='+str(score))
    L.append(score)
    
print('mean column-wise ROC AUC:', sum(L)*1.0/len(L))

end=time.time()

print(end-start)
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = classes)], axis = 1)
submission.to_csv('submission.csv', index=False)