# Load Libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Clean
import re
import nltk
import ftfy
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import SnowballStemmer
from nltk import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from langdetect import detect

# Vectorizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def main():
    print("This module contains functions for cleaning and Pre-Processing Reddit posts")
    
def clean_posts(posts_ls):
    '''Removes URLs, usernames, subreddit names, whitespace, deleted or removed posts and fixed non UTF-8 encoded text'''
    posts_clean = []
    URL_format = re.compile("http[s]{0,1}://\S+", re.UNICODE)
    username_format = re.compile("@[A-Za-z_0-9]+", re.UNICODE)
    subreddit_format = re.compile(r"/r/\S+", re.UNICODE)
    for post in posts_ls:
        # Remove URLs
        post_noURL = re.sub(URL_format, "", post)
        # Remove subreddit references
        post_noSubreddit = re.sub(subreddit_format, '', post_noURL)
        # Replace user mentions with @
        post_noMention = re.sub(username_format, "", post_noSubreddit)
        # Slash
        post_noSlash = post_noMention.replace('\n', ' ').replace("\\", "")
        # Deleted 
        post_noDel = post_noSlash.replace("[deleted]", "").replace("[removed]", "")
        # Fix weirdly encoded texts (like &amp)
        post = ftfy.fix_text(post_noDel)
           
        posts_clean.append(post)

    return posts_clean

def delete_bots_mods(df):
    '''Removes Moderator users based on author name and author flair text'''
    df = df[(df['author']!='AutoModerator')&(df['author']!='[deleted]')]
    df = df[(df['author_flair_text']!='Moderator')&(df['author_flair_text']!='mod')]
    return df
    
def mod_keywords(df):
    '''Removes posts that contain moderator keywords'''
    index_ls = []
    mod_words = ['moderator', "moderators", "mod", "mods", "admin", "admins", "welcome", 'removed', 'deleted', "guidelines"]
    for index, post in enumerate(df['text_clean']):
        word_tokens = nltk.word_tokenize(post.lower())
        for word in word_tokens:
            if word in mod_words:
                index_ls.append(index)
    
    index_ls_unique = list(set(index_ls))
    df = df.drop(df.index[index_ls_unique])
    return df

def utc_time_convert(date_ls):
    '''Convert UTC Timestamp to Datetime format'''
    new_date = [datetime.utcfromtimestamp(date) for date in date_ls]
    return new_date
    
def expandContractions(text, c_re=c_re):
    """Expand contractions on common English contractions"""
    contraction_dictionary = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because"
     , "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not"
     , "doesn't": "does not", "don't": "do not", "gonna": "going to", "hadn't": "had not", "hadn't've": "had not have"
     , "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will"
     , "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will"
     , "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am"
     , "i've": "i have", "isn't": "is not", "it'd": "it had", "it'd've": "it would have", "it'll": "it will"
     , "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not"
     , "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have"
     , "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have"
     , "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not"
     , "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have"
     , "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have"
     , "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that would"
     , "that'd've": "that would have", "that's": "that is", "there'd": "there had", "there'd've": "there would have"
     , "there's": "there is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will"
     , "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not"
     , "we'd": "we had", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are"
     , "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are"
     , "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did"
     , "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is"
     , "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not"
     , "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have"
     , "y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would", "y'all'd've": "you all would have"
     , "y'all're": "you all are", "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have"
     , "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "'m":"i am"
    }    
    
    c_re = re.compile('(%s)' % '|'.join(contraction_dictionary.keys()))
    def replace(match):
        return contraction_dictionary[match.group(0)]
    return c_re.sub(replace, post)

def gridsearch_preprocessing(posts, remove_hash_emoji=True, lowercase=True
                            , expand_contractions=True, remove_punctuation = True):
    posts_clean = [] 
    for post in posts:
        
        post = post.replace("’", "'")
        
        # Hashtag and Emoji Removal
        if remove_hash_emoji == True:
            post = ' '.join(re.sub("(\#[A-Za-z0-9]+)|(<Emoji:.*>)", " ", post).split())
        
        # Lowercase
        if lowercase == True:    
            post = post.lower()
        
        # Replace Contractions
        if expand_contractions == True:
            post = expandContractions(post)
        
        # Punctuation Removal
        if remove_punctuation == True:    
            post = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", post).split())

        posts_clean.append(post)
    return posts_clean

def stopwords_pronoun(posts_ls, exclude_pronoun = False):
    '''Removes standard list of stopwords with the option to include pronouns'''
    posts_stopwords = [] 
    stop_words = list(set(stopwords.words('english')))
    pronoun_words = ['i', "he", 'him', 'himself', 'his', "she", "her", "hers", "herself", "me", "myself", "mine", "ours"
                     , "ourselves", "we", "us", "they", "themselves", "their", "theirs", "you", "your", "yours", "yourself"
                     , "yourselves"]
    stop_words_pronoun = [word for word in stop_words if word not in pronoun_words]
    # Stopwords Removal
    for post in posts_ls:
        word_tokens = nltk.word_tokenize(post) 
        # Exclude Stopwords
        if exclude_pronoun == False:
            cleaned_sentence = [word for word in word_tokens if not word in stop_words]
        # Include Stopwords
        else: 
            cleaned_sentence = [word for word in word_tokens if not word in stop_words_pronoun]
        post = ' '.join(cleaned_sentence)
        posts_stopwords.append(post)

    return posts_stopwords


def stemmer_lemmatiser(posts_ls, porter=True, snowball=False, lancaster=False, lemma = False):
    '''Stems or Lemmatises posts'''
    posts_clean = [] 
    
    if lemma == True:
        wordnet_lemmatizer = WordNetLemmatizer()

    for post in posts_ls:
        word_tokens = nltk.word_tokenize(post) 
        if porter == True:
            post = ' '.join([PorterStemmer().stem(word) for word in word_tokens])     
        if snowball == True:
            post = ' '.join([SnowballStemmer('english').stem(word) for word in word_tokens])
        if lancaster == True:
            post = ' '.join([LancasterStemmer().stem(word) for word in word_tokens]) 
        if lemma == True:
            post = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])

    # Append 
    posts_clean.append(post)
    
    return posts_clean

def vectoriser(posts_ls, word=True, count=True, min_df=0.01, max_df=0.5, ngram=1, max_features = 50000):
    '''Applies CountVectorise or TfidfVectoriser to posts to analyse word n-grams or character n-grams'''
    
    # Word n-grams
    if (count == True) and (word == True):
        vec = CountVectorizer(analyzer = 'word', tokenizer = tokenize, min_df = min_df, max_df = max_df
                              , ngram_range = (ngram, ngram), max_features = max_features)
        
    if (count == False) and (word == True):
        vec = TfidfVectorizer(analyzer = 'word', tokenizer = tokenize, min_df = min_df, max_df = max_df
                              , ngram_range = (ngram, ngram), max_features = max_features)
        
    ### Character n-grams
    if (count == True) and (word == False):
        vec = CountVectorizer(analyzer = 'character', tokenizer = tokenize, min_df = min_df, max_df = max_df
                              , ngram_range = (ngram, ngram), max_features = max_features)
        
    if (count == True) and (word == True):
        vec = TfidfVectorizer(analyzer = 'character', tokenizer = tokenize, min_df = min_df, max_df = max_df
                              , ngram_range = (ngram, ngram), max_features = max_features)
               
    # Apply Vectorisation to Training and Testing Data
    posts_vec = vec.fit_transform(posts_ls).toarray()
    
    return posts_vec

if __name__ == '__main__':   
    main()

