import pandas as pd
import re
import nltk


def main():
    print("This module contains functions for scraping PushShift for Reddit data")
    
def punctuation(posts_ls):
    """Provide count of punctuation usage"""
    uppercase = [len(re.findall(r'[A-Z]', post)) for post in posts_ls]
    period = [post.count('.') for post in posts_ls]
    exclamation = [post.count('!') for post in posts_ls]
    question = [post.count('?') for post in posts_ls]
    
    return uppercase, period, exclamation, question


def liwc_features(df, l):
    '''Create LIWC features for posts'''
    
    def extract_emotion(l):
        '''Creates a dictionary and extract different LIWC features out of key words in posts'''
        dic = {'i':0, 'we':0, 'you':0, 'shehe':0, 'they':0, 'posemo':0
             , 'negemo':0, 'sad':0, 'anger':0, 'anx':0, 'family':0
             , 'friend':0, 'negate':0, 'swear':0, 'cogmech':0, 'insight':0
             , 'cause':0, 'discrep':0, 'tentat':0, 'certain':0, 'inhib':0
             , 'incl':0, 'excl':0, 'feel':0, 'health':0, 'sexual':0
             , 'time':0, 'work':0, 'home':0, 'money':0, 'relig':0
             , 'death':0, "past":0, "present":0, 'future':0}
        for w in l:
            for emotion in liwc.search(w):
                if emotion in dic.keys():
                    dic[emo] += 1
        return dic

    def tokenize(text):
        '''Tokenises the post'''
        tokens = nltk.word_tokenize(text)
        return tokens
    
    df['tokens'] = df.text_preprocessed.apply(tokenize)
    df['emo_dict'] = df.tokens.apply(extract_emotion)
    fps = df.emo_dict.apply(lambda x: x['i'])
    fpp = df.emo_dict.apply(lambda x: x['we'])
    sp = df.emo_dict.apply(lambda x: x['you'])
    tps = df.emo_dict.apply(lambda x: x['shehe'])
    tpp = df.emo_dict.apply(lambda x: x['they'])
    pos = df.emo_dict.apply(lambda x: x['posemo'])
    neg = df.emo_dict.apply(lambda x: x['negemo'])
    sadness = df.emo_dict.apply(lambda x: x['sad'])
    anger = df.emo_dict.apply(lambda x: x['anger'])
    anxiety = df.emo_dict.apply(lambda x: x['anx'])
    family = df.emo_dict.apply(lambda x: x['family'])
    friend = df.emo_dict.apply(lambda x: x['friend'])
    negation = df.emo_dict.apply(lambda x: x['negate'])
    swear = df.emo_dict.apply(lambda x: x['swear'])
    cognitivemech = df.emo_dict.apply(lambda x: x['cogmech'])
    insight = df.emo_dict.apply(lambda x: x['insight'])
    causation = df.emo_dict.apply(lambda x: x['cause'])
    discrep = df.emo_dict.apply(lambda x: x['discrep'])
    tentative = df.emo_dict.apply(lambda x: x['tentat'])
    certainty = df.emo_dict.apply(lambda x: x['certain'])
    inhibit = df.emo_dict.apply(lambda x: x['inhib'])
    inclusive = df.emo_dict.apply(lambda x: x['incl'])
    exclusive = df.emo_dict.apply(lambda x: x['excl'])
    affective = df.emo_dict.apply(lambda x: x['feel'])
    health = df.emo_dict.apply(lambda x: x['health'])
    sexual = df.emo_dict.apply(lambda x: x['sexual'])
    time = df.emo_dict.apply(lambda x: x['time'])
    work = df.emo_dict.apply(lambda x: x['work'])
    home = df.emo_dict.apply(lambda x: x['home'])
    money = df.emo_dict.apply(lambda x: x['money'])
    religion = df.emo_dict.apply(lambda x: x['relig'])
    death = df.emo_dict.apply(lambda x: x['death'])
    past = df.emo_dict.apply(lambda x: x['past'])
    present = df.emo_dict.apply(lambda x: x['present'])
    future = df.emo_dict.apply(lambda x: x['future'])
    
    return fps, fpp, sp, tps, tpp, pos, neg, sadness, anger, anxiety, family, friend, negation, swear, cognitivemech, insight
           , causation, discrep, tentative, certainty, inhibit, inclusive, exclusive, affective, health, sexual, time, work
           , home, money, religion, death, past, present, future

def create_ratios(metric_ls, length_ls):
    """Normalise features by dividing over a specific length (e.g. total characters in a post for punctuation"""
    ratio = [metric/length for metric, length in zip(metric_ls, length_ls)]

    return ratio

def helpseeking_variable(posts_ls):
    '''For Reddit dataset, creates a help-seeking variable using the depression help-seeking lexicon'''
    
    # Load DHSL
    helpseek_words = pd.read_csv("helpseek_words.csv")
    helpseek_dict = dict(zip(helpseek_words['helpseek_words'],helpseek_words['helpseek_chi']))
    
    
    helpseek_chi_ls = []
    for post in posts_ls:
        helpseek_chi = 0
        words = i.split()
        for word in words:
            if word in helpseek_dict.keys():
                helpseek_chi += helpseek_dict[word]
        prop_helpseek_chi = helpseek_chi/len(words)
        helpseek_chi_ls.append(round(prop_helpseek_chi, 2))
        
    return helpseek_chi_ls

if __name__ == '__main__':   # Only executed if it is run as a script
    main()

