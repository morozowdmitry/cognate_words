import pymorphy2


class Lemmatizer(object):
    """Class for token lemmatization using PyMorphy (returning most probable)."""

    def __init__(self, *args, **kwargs):
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    def lemmatize(self, token):
        return self.lemmatizer.parse(token)[0]
