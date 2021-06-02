import os
import re

import nltk
import pandas as pd

from utils import clean_html


DEFAULT_DATASET_NAME = 'all_bills.pkl'
MIN_LENGTH = 2000
MIN_YEAR = 1990

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#_]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
NUMBERS = re.compile('[\b\d+\b]')


def clean_bill_text(text, lemmatizer, stopwords, law_stopwords):
    text = text.lower()  # Lowercase
    text = clean_html(text)  # Clean text from html tags
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # Unify indents
    text = re.sub(BAD_SYMBOLS_RE, " ", text)  # Clean from other bad symbols
    text = re.sub(NUMBERS, " ", text)  # Delete from numbers
    text = text.split()

    # Take only the relevant part of the bills, ignoring an introduction part
    try:
        text = text[text.index("representatives") + 4:]
    except ValueError:
        pass

    text = ' '.join([lemmatizer.lemmatize(word) for word in text if
                     word not in stopwords and word not in law_stopwords and len(word) > 1])
    return text


def clean_dataset(path_to_data):
    all_bills = pd.read_pickle(os.path.join(path_to_data, DEFAULT_DATASET_NAME))

    all_bills = all_bills[(all_bills['length'] > MIN_LENGTH) & (all_bills['year'] >= MIN_YEAR)]
    all_bills = all_bills[['conc', 'text', 'lobbied']]

    # Load stopwords
    STOPWORDS = nltk.corpus.stopwords.words('english')
    LAW_STOPWORDS = pd.read_csv(os.path.join(path_to_data, 'stopwords.csv'), header=None)[0].values.tolist()
    LAW_STOPWORDS = [word.strip() for word in LAW_STOPWORDS]

    lemmatizer = nltk.stem.WordNetLemmatizer()

    all_bills['text'] = all_bills['text'].apply(lambda x: clean_bill_text(x, lemmatizer,
                                                                          STOPWORDS, LAW_STOPWORDS))
    all_bills.columns = ['b_id', 'text', 'lobbied']
    all_bills.to_pickle('better_cleaned_pickles_with_id.pickle')

