
import re
import numpy as np
from numba import jit
from collections import deque

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

@jit(nopython=True)
def normalized(arr):
  total = arr.sum()
  return arr / total

def next_line_with_rotation(f):
  s = f.readline()
  if len(s)==0:
    f.seek(0)
    s = f.readline()
  return s

def data_generator(corpus_path, window_sizes, post_func):
  max_window = max(window_sizes)
  length = 1 + 2*max_window
  with open(corpus_path, 'r', encoding='utf-8') as f:
    # Initialize buffer and window
    buffer = deque([])
    window = deque([])
    while len(buffer) < length * 10 + max_window:
      buffer.extend(next_line_with_rotation(f).lower())
    for i in range(length):
      window.append(buffer.popleft())
    assert len(window) == length
    #
    while True:
      # extend buffer if needed
      while len(buffer) < length*10:
        buffer.extend(next_line_with_rotation(f).lower())
      #
      window.popleft()
      window.append(buffer.popleft())
      yield post_func(window)
