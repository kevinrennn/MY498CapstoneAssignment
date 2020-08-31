from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models.phrases import Phrases, Phraser

def main():
    print("This module contains functions for word embedding")
    
def similar_word_vectors(posts_ls):
    """Provide count of punctuation usage"""
    # Create sentences for W2V model; keep bigrams
    sent = [row.split() for row in posts_ls]
    phrases = Phrases(sent, min_count=5, progress_per=20000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    
    # Train model
    w2v_model = Word2Vec(sentences, 
                         min_count=5,
                         window=5,
                         size=200,
                         sg=1,
                         workers=3)

    # Obtain word vectors
    word_vectors = w2v_model.wv
    
    most_similar_vectors = word_vectors.most_similar('professional_help', topn=500)
    
    return most_similar_vectors

if __name__ == '__main__':   # Only executed if it is run as a script
    main()

