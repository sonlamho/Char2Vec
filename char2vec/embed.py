
import numpy as np
import tensorflow as tf
from .utils import Tokenizer, ALPHABET, normalized, next_line_with_rotation
from collections import deque
from functools import partial

class CONFIG:
  """Model hyperparams"""
  D = 10          # embedding dimension
  WINDOW_SIZES = [1,2,3]
  BATCH = 32
  SHUFF_BUFFER = 10000
  TOTAL_STEPS = 30000
  GPU = True

class Char2Vec(object):

  def __init__(self, config=CONFIG, alphabet=ALPHABET, unk='~', DTYPE=tf.float32):
    self.cfg = config
    self.graph = tf.Graph()
    self.DTYPE = DTYPE
    self.tokenizer = Tokenizer(alphabet, unk)
    self.V = self.tokenizer.V
    self.N_HEADS = 2 * len(config.WINDOW_SIZES)
    self.__graph_created = False

  def create_graph(self, corpus_path):
    with self.graph.as_default():
      self.batch_size = tf.placeholder_with_default(
        tf.constant(self.cfg.BATCH, dtype=tf.int64), shape=[], name='batch_size')
      dataset = tf.data.Dataset().from_generator(
        self.data_generator,
        (self.DTYPE, self.DTYPE),
        (tf.TensorShape([self.V]), tf.TensorShape([self.N_HEADS*self.V])),
        args=[corpus_path, self.cfg.WINDOW_SIZES]
      )
      dataset = dataset.shuffle(self.cfg.SHUFF_BUFFER)
      dataset = dataset.batch(self.batch_size)
      self.dataset = dataset.prefetch(10)
      self.data_iter = self.dataset.make_initializable_iterator()
      self.x_in, self.y_labels = self.data_iter.get_next()

      device = '/device:GPU:0' if self.cfg.GPU else '/cpu:0'
      with self.graph.device(device):
        self.U = tf.get_variable(dtype=self.DTYPE, shape=[self.V, self.cfg.D], name='U')
        self.rep = tf.matmul(self.x_in, self.U)   # shape [batch_size, D]
        self.W = tf.get_variable(dtype=self.DTYPE,
                               shape=[self.cfg.D, self.N_HEADS*self.V],
                               name='W')
        self.logits = tf.matmul(self.rep, self.W)  # shape [batch_size, N*V]

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=self.y_labels, logits=self.logits
        ))
        self._optimizer = tf.train.AdamOptimizer()
        self.train_step = self._optimizer.minimize(self.loss)
        self.__graph_created = True

  def data_generator(self, corpus_path, window_sizes):
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
        yield self._xy_arrays(window, midpos=max_window)

  def _xy_arrays(self, window, midpos):
    X = self.tokenizer.to_1hot(window[midpos]).flatten()  #length V
    Ys = []
    for p in self.cfg.WINDOW_SIZES:
      t_left = window[midpos - p]
      t_right = window[midpos + p]
      Ys.append(self.tokenizer.to_1hot(t_left).flatten())
      Ys.append(self.tokenizer.to_1hot(t_right).flatten())
    Y = np.concatenate(Ys)
    return X, Y

  def fit(self, path_to_corpus):

    pass #TODO


class CharGloVe(object):
  # TODO
  pass
