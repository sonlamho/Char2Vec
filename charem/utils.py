
import re
import numpy as np
from numba import jit

ALPHABET = """abcdefghijklmnopqrstuvwxyz1234567890,.()[]"' -\n"""

class Tokenizer(object):

  def __init__(self, alphabet=ALPHABET, unk='~'):
    assert unk not in alphabet, "please keep UNK character not part of alphabet"
    self.alphabet = sorted(list(set(alphabet)))
    self._to_int = {  # dict mapping known characters to integers
      c:i for i,c in enumerate(self.alphabet)
    }
    self._unk_i = len(self.alphabet)  # int for unknown chars
    self.unk = unk
    self.V = len(self.alphabet) + 1   # dim of 1-hot array

  def to_ints(self, text):
    """Returns a list of integers"""
    return [self._to_int.get(c, self._unk_i) for c in text]

  def to_1hot(self, text):
    """Return an array of shape `(len(text), len(self.alphabet)+1)`."""
    tokens = self.to_ints(text)
    arr = np.zeros(shape=(len(text), len(self.alphabet)+1), dtype='float32')
    # each row of arr represent a character of text
    for i_row, tok in enumerate(tokens):
      arr[i_row, tok] = 1
    return arr

  def from_ints(self, L):
    """Reconstruct the text given list of integer tokens"""
    return ''.join([
      self.alphabet[i] if i < len(self.alphabet) else self.unk for i in L ])

  def from_1hot(self, arr):
    """Reconstruct the text given an array"""
    return self.from_ints([row.argmax() for row in arr])

def normalized(arr):
  total = arr.sum()
  return arr / total

def next_line_with_rotation(f):
  s = f.readline()
  if len(s)==0:
    f.seek(0)
    s = f.readline()
  return s
