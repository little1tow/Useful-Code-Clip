import re
import BeautifulSoup
import unicodedata
import contractions
import nlp


from pyspellchecker import SpellChecker


import nltk
from nltk import stpwrds

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)
    return text


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    # It will expand shortened words, For example : don’t to do not & wanna to want to.
    text = contractions.fix(text)
    return text


def word_to_num(text):
    doc = nlp(text)
    tokens = [w2n.word_to_num(token.text) if token.pos_ == 'NUM' else token for token in doc]
    tokens = " ".join([str(tok) for tok in tokens])
    return tokens


def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


## Spelling correction library ( pip install pyspellchecker)
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token.lower() not in stpwrds]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stpwrds]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


"""Main function to apply all above cleaning fucntions with adjustable parameters to be passed"""
def normalize_doc(doc, URL_stripping=True, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True , emoji_removal=True,
                     spelling_correction = True, word_to_num=False):
    normalized_document = []
    # Stip URL's
    if URL_stripping:
        doc = remove_URL(doc)
    # strip HTML
    if html_stripping:
        doc = strip_html_tags(doc)
    # remove accented characters
    if accented_char_removal:
        doc = remove_accented_chars(doc)
    # expand contractions    
    if contraction_expansion:
        doc = expand_contractions(doc)
    # lowercase the text    
    if text_lower_case:
        doc = doc.lower()
    # Word to numbers
    if word_to_num:
        doc = word_to_num(doc)
    # remove extra newlines
    doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
    # remove special characters    
    if special_char_removal:
        doc = remove_special_characters(doc)
    # remove extra whitespace
    doc = re.sub(' +', ' ', doc)
    # remove emogis
    if emoji_removal:
        doc = remove_emoji(doc)
    # spelling_correction
    if spelling_correction:
        doc = correct_spellings(doc)
    # lemmatize text
    if text_lemmatization:
        doc = lemmatize_text(doc)
    # remove stopwords
    if stopword_removal:
        doc = remove_stopwords(doc, is_lower_case=text_lower_case)
    normalized_document.append(doc)
    return doc





