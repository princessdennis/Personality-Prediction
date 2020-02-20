import re
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

class dineise_clean_text(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        def clean_text(row):
            # Take out website links in each row
            row = re.sub(r'(?<=.)*https?:\/\/[a-zA-Z0-9\/\.\?\-=_]+(?=.*)', '', row)  # Take out website links
            row = re.sub(r'(?<=.)*[a-zA-Z0-9\/\.\?\-=_]+(jpg|png|gif|html|php)(?=.*)', '',
                         row)  # Take out png, jpg, gif, etc.. links

            # Replace some apostrophes
            replace_list = {r"i'm": 'i am', r"'re": ' are', r"let’s": 'let us',
                            r"'s": ' is', r"'ve": ' have', r"can't": 'can not',
                            r"cannot": 'can not', r"shan’t": 'shall not', r"n't": ' not',
                            r"'d": ' would', r"'ll": ' will', r"'scuse": 'excuse', '\s+': ' '}
            row = row.lower()  # lowercase all words in sentence
            # Iterate through the replacement list and replace any match
            for rep_word in replace_list:
                row = row.replace(rep_word, replace_list[rep_word])

            # Replace any character that's NOT a letter with a space
            row = re.sub(r"[^A-Za-z]+", ' ', row)

            # Split words into a list based on delimiters \s and \|
            row_split = re.split(r'[\s\|]', row)

            # Stem each word in row
            # **Could try SnowballStemmer('english')
            lemmatizer = WordNetLemmatizer()
            new_row = ''
            for word in row_split:
                new_word = lemmatizer.lemmatize(word)
                new_row = new_row + ' ' + new_word

            return new_row

        X1 = X.apply(lambda x: clean_text(x))
        return X1




# Don't need the code below, but leave them just in case there's an inheritance problem later
# if __name__=='__main__':
#     X1 = dineise_clean_text()
#     dineise_clean_text.__module__ = "predict_api"