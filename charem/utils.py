
import re
import numpy as np

ALPHABET = """abcdefghijklmnopqrstuvwxyz1234567890,.()[]"' -\n"""

class Tokenizer(object):

  def __init__(self, alphabet=ALPHABET):
    self.alphabet = sorted(list(set(alphabet)))
    self._to_int = {  # dict mapping known characters to integers
      c:i for i,c in enumerate(self.alphabet)
    }
    self._unk_i = len(self.alphabet)  # int for unknown chars

  def to_ints(self, text):
    """Returns a list of integers"""
    return [self._to_int.get(c, self._unk_i) for c in text]

  def to_1hot(self, text):
    """Return an array of shape `(len(text), len(self.alphabet)+1)`."""
    tokens = self.to_ints(text)
    arr = np.zeros(shape=(len(text), len(self.alphabet)+1), dtype='float32')
    for pos, tok in enumerate(tokens):
      arr[pos, tok] = 1
    return arr

  def from_ints(self, L):
    """Reconstruct the text given list of integer tokens"""
    # TODO:
    pass
