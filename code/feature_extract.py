import pandas as pd
import numpy as np
import spacy
import readability
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag, pos_tag_sents


# Taken this from https://www.kaggle.com/ravishah1/readability-feature-engineering-non-nn-baseline
def readability_measurements(passage: str):
    """
    This function uses the readability library for feature engineering.
    It includes textual statistics, readability scales and metric, and some pos stats
    """
    results = readability.getmeasures(passage, lang='en')
    
    chars_per_word = results['sentence info']['characters_per_word']
    syll_per_word = results['sentence info']['syll_per_word']
    words_per_sent = results['sentence info']['words_per_sentence']
    
    kincaid = results['readability grades']['Kincaid']
    ari = results['readability grades']['ARI']
    coleman_liau = results['readability grades']['Coleman-Liau']
    flesch = results['readability grades']['FleschReadingEase']
    gunning_fog = results['readability grades']['GunningFogIndex']
    lix = results['readability grades']['LIX']
    smog = results['readability grades']['SMOGIndex']
    rix = results['readability grades']['RIX']
    dale_chall = results['readability grades']['DaleChallIndex']
    
    tobeverb = results['word usage']['tobeverb']
    auxverb = results['word usage']['auxverb']
    conjunction = results['word usage']['conjunction']
    pronoun = results['word usage']['pronoun']
    preposition = results['word usage']['preposition']
    nominalization = results['word usage']['nominalization']
    
    pronoun_b = results['sentence beginnings']['pronoun']
    interrogative = results['sentence beginnings']['interrogative']
    article = results['sentence beginnings']['article']
    subordination = results['sentence beginnings']['subordination']
    conjunction_b = results['sentence beginnings']['conjunction']
    preposition_b = results['sentence beginnings']['preposition']

    
    return [chars_per_word, syll_per_word, words_per_sent,
            kincaid, ari, coleman_liau, flesch, gunning_fog, lix, smog, rix, dale_chall,
            tobeverb, auxverb, conjunction, pronoun, preposition, nominalization,
            pronoun_b, interrogative, article, subordination, conjunction_b, preposition_b]

def spacy_features(df: pd.DataFrame):
    """
    This function generates features using spacy en_core_wb_lg
    I learned about this from these resources:
    https://www.kaggle.com/konradb/linear-baseline-with-cv
    https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
    """
    
    nlp = spacy.load('en_core_web_lg')
    with nlp.disable_pipes():
        vectors = np.array([nlp(text).vector for text in df.excerpt])
        
    return vectors

def get_spacy_col_names():
    names = list()
    for i in range(300):
        names.append(f"spacy_{i}")
        
    return names

def pos_tag_features(passage: str):
    """
    This function counts the number of times different parts of speech occur in an excerpt
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]
    
    tags = pos_tag(word_tokenize(passage))
    tag_list= list()
    
    for tag in pos_tags:
        tag_list.append(len([i[0] for i in tags if i[1] == tag]))
    
    return tag_list

def generate_other_features(passage: str):
    """
    This function is where I test miscellaneous features
    This is experimental
    """
    # punctuation count
    periods = passage.count(".")
    commas = passage.count(",")
    semis = passage.count(";")
    exclaims = passage.count("!")
    questions = passage.count("?")
    
    # Some other stats
    num_char = len(passage)
    num_words = len(passage.split(" "))
    unique_words = len(set(passage.split(" ") ))
    word_diversity = unique_words/num_words
    
    word_len = [len(w) for w in passage.split(" ")]
    longest_word = np.max(word_len)
    avg_len_word = np.mean(word_len)
    
    return [periods, commas, semis, exclaims, questions,
            num_char, num_words, unique_words, word_diversity,
            longest_word, avg_len_word]

def extract_features(df):

    scores_df = pd.DataFrame(df["excerpt"].apply(lambda p : readability_measurements(p)).tolist(), 
                                 columns=["chars_per_word", "syll_per_word", "words_per_sent",
                                          "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog", "rix", "dale_chall",
                                          "tobeverb", "auxverb", "conjunction", "pronoun", "preposition", "nominalization",
                                          "pronoun_b", "interrogative", "article", "subordination", "conjunction_b", "preposition_b"])
    df = pd.merge(df, scores_df, left_index=True, right_index=True)
    
    spacy_df = pd.DataFrame(spacy_features(df), columns=get_spacy_col_names())
    df = pd.merge(df, spacy_df, left_index=True, right_index=True)
    
    pos_df = pd.DataFrame(df["excerpt"].apply(lambda p : pos_tag_features(p)).tolist(),
                            columns=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", 
                                    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                                    "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"])
    df = pd.merge(df, pos_df, left_index=True, right_index=True)
    
    other_df = pd.DataFrame(df["excerpt"].apply(lambda p : generate_other_features(p)).tolist(),
                            columns=["periods", "commas", "semis", "exclaims", "questions",
                                        "num_char", "num_words", "unique_words", "word_diversity",
                                        "longest_word", "avg_len_word"])
    df = pd.merge(df, other_df, left_index=True, right_index=True)

    return df