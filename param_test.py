"""
This module was for figuring out which parameters actually contribute to the effectiveness of the module.
I iterated through 4 parameters and looked at what changed the score when all other params. were fixed.

    Conclusions (min_df=5, varied stopwords english/none):
    * C value for log. regression does very little to improve model quality.
      One of the explanations for the iteration warning (did not converge) in the notebook hinted that there might be good seperators 
      in the features which might explain why varying C did very little. (L2 is enabled by default without tampering with C)
    * tfidf vectorizer does improve the quality a bit because it "normalises" the data and 
      therefore the model has better parameter fits.
    * Use of english stopwords actually reduces the accuracy. 
      Might have something to do with how sarcastic comments are shorter and we lose information by ditching them (Fig. 4 in the publication)
    * n-gram sizes above 1-2 reduce the accuracy. Probably due to overfitting. 
      Miht have to push down the amount of features. Mby increasing min_df?
    
Tried increasing mind_df to 6, got basically the same results.
The conclusion might be that the model suffers not from too many irrelevant features but from too many specific ones.
    
"""
import pandas as pd
import os
import datetime
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer



def model(ngr_range,vect_type):
    vectstart = time.time()
    print("vectorising the data set with ",vect_type,"...")
    # min_df will be kept to a 5 regardless
    if vect_type == 'countvect':
        vect = CountVectorizer(min_df=6, ngram_range=ngr_range)
    if vect_type == 'tfidf':
        vect = TfidfVectorizer(min_df=6, ngram_range=ngr_range)
    vect.fit(test_set['comment'])
    print(time.time()-vectstart,"seconds elapsed\n\n")

    X_tr, X_tst, y_train, y_test = train_test_split(test_set['comment'],test_set['label'])

    vectfitstart = time.time()
    print("Fitting the data on the matrix...")
    X_train = vect.transform(X_tr)
    X_test = vect.transform(X_tst)
    print(time.time() - vectfitstart," seconds elapsed\n\n")

    feature_names = vect.get_feature_names()
    print(feature_names[::2000])
    print(len(feature_names))

    print("fitting the model...")
    clf = LogisticRegression()
    shape = X_train.shape
    for C_value in [0.1, 1, 10]:
        fitstart = time.time()
        clf.fit(X_train,y_train)
        print("\n",time.time() - fitstart, " seconds elapsed for C = ",C_value," \n")

        score = clf.score(X_test,y_test)

        content = "score: " + str(score) + "\n" + "shape: "+ str(shape)
        print("ngr_range: ",ngr_range, " vect_type: ", vect_type, " stopwords: None "," min_df = 6")
        print("score: ", score)
        print(shape,"\n")

train_data = pd.read_csv('data/train.csv')
test_set = train_data.dropna()

for lower_ngram_bound in [1,2]:
    for upper_ngram_bound in [2,3]:
        for vectorization_type in ['countvect','tfidf']:
            model(ngr_range=(lower_ngram_bound,upper_ngram_bound),vect_type=vectorization_type)
