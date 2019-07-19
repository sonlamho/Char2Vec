
import numpy as np
import tensorflow as tf
from .utils import Tokenizer, ALPHABET, normalized

class CONFIG:
  """Model hyperparams"""
  D = 20          # embedding dimension

  # left and right window size
  L_WINDOW = 2
  R_WINDOW = 2

  BATCH = 128
  EPOCHS = 30


class Char2Vec(object):

  N_HEADS = 3

  def __init__(self, config=CONFIG, alphabet=ALPHABET, unk='~', DTYPE='float32'):
    self.cfg = config
    self.graph = tf.Graph()
    self.DTYPE = DTYPE
    self.tokenizer = Tokenizer(alphabet, unk)
    self.V = self.tokenizer.V

  def create_graph(self):
    with self.graph.as_default():
      self.x_in = tf.placeholder(dtype=self.DTYPE, shape=[None, self.V], name='x_in')  #  batch_size will replace None in shape
      self.U = tf.get_variable(dtype=self.DTYPE, shape=[self.V, self.cfg.D], name='U')
      self.rep = tf.matmul(self.x_in, self.U)   # shape [batch_size, D]
      self.Ws, self.logits = self._construct_logits(self.rep, self.N_HEADS)
      #self.y_hats = [tf.nn.softmax(t, axis=-1) for t in self.logits]
      self.y_labels = [
        tf.placeholder(dtype=self.DTYPE, shape=[None, self.V])
        for i in range(self.N_HEADS)
      ]
      self.losses = [
        tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=label, logits=logit)
        for logit, label in zip(self.logits, self.y_labels)
      ]
      self.loss = tf.add_n([tf.reduce_mean(t) for t in self.losses])
      self._optimizer = tf.train.AdamOptimizer()
      self.train_step = self._optimizer.minimize(self.loss)

  def _construct_logits(self, rep, n_heads):
    """ """
    Ws = []
    logits = []
    with self.graph.as_default():
      for i in range(n_heads):
        weights_i = tf.get_variable(dtype=self.DTYPE, shape=[self.cfg.D, self.V], name='W_'+str(i))
        logits_i = tf.matmul(rep, weights_i)
        Ws.append(weights_i)
        logits.append(logits_i)
    return Ws, logits

  def fit(self, path_to_corpus):
    pass #TODO


class CharGloVe(object):
  # TODO
  pass
