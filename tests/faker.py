from __future__ import unicode_literals, print_function

from builtins import range

import random
import string

import numpy as np
import pandas as pd


def _id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


_RANDOM_LABELS = [_id_generator() for _ in range(30)]

_RANDOM_NOUNS = ("puppy", "car", "rabbit", "girl", "boy", "house", "monkey", "donkey")
_RANDOMS_VERBS = ("runs", "hits", "jumps", "drives", "barfs", "eats", "swims", "weeps")
_RANDOM_ADVERBS = ("crazily", "dutifully", "foolishly", "merrily", "occasionally", "crazily")
_RANDOM_ADJECTIVES = ("adorable", "clueless", "dirty", "odd", "stupid")
_SENTENCE_ELEMENTS = [_RANDOM_NOUNS, _RANDOMS_VERBS, _RANDOM_ADJECTIVES, _RANDOM_ADVERBS]


def _get_random_sentence():
    return ' '.join([random.choice(i) for i in _SENTENCE_ELEMENTS])


def get_random_dataframe(size):
    df = pd.DataFrame(np.random.randn(size, 4), columns=list('ABCD'))

    # Label column
    df['B'] = [random.choice(_RANDOM_LABELS) for _ in range(len(df))]

    # Label list column
    df['C'] = [[random.choice(_RANDOM_LABELS) for __ in range(random.randint(0, 4))] for _ in range(len(df))]

    # Text column
    df['D'] = [_get_random_sentence() for _ in range(len(df))]

    return df
