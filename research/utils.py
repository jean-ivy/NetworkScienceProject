from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(X_train, X_test, ngram_range=(1, 2)):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=0.9, min_df=15, token_pattern='(\S+)')
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    return X_train, X_test, tfidf_vectorizer


def clean_html(raw_html):
    return BeautifulSoup(raw_html, "lxml").text