
import numpy as np
import tensorflow as tf
from .utils import Tokenizer, ALPHABET, normalized, next_line_with_rotation, data_generator

class CONFIG:
  """Model hyperparams"""
  D = 10          # embedding dimension
  WINDOW_SIZES = [1,2,3]
  BATCH = 32
  SHUFF_BUFFER = 10000
  TOTAL_STEPS = 30001
  GPU = False

  @classmethod
  def show(cls):
    print('Showing {}'.format(cls.__name__))
    for a in dir(cls):
      if not a.startswith('_'):
        ob = getattr(cls, a)
        if not callable(ob):
          print('  {} = {}'.format(a, ob))


class Char2Vec(object):

  def __init__(self, corpus_path, config=CONFIG, alphabet=ALPHABET, unk='~', DTYPE=tf.float32):
    """"""
    self._corpus_path = corpus_path
    self._cfg = config
    self._DTYPE = DTYPE
    self._tokenizer = Tokenizer(alphabet, unk)
    self._V = self._tokenizer.V
    self._N_HEADS = 2 * len(config.WINDOW_SIZES)
    self._graph = None

  def _create_graph(self):
    """Create the computational graph, save graph and some nodes as instance variables."""
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._batch_size = tf.placeholder_with_default(
        tf.constant(self._cfg.BATCH, dtype=tf.int64), shape=[], name='batch_size')
      dataset = tf.data.Dataset().from_generator(
        self._data_generator,
        (self._DTYPE, self._DTYPE),
        (tf.TensorShape([self._V]), tf.TensorShape([self._N_HEADS*self._V])),
        args=[self._corpus_path, self._cfg.WINDOW_SIZES]
      )
      dataset = dataset.shuffle(self._cfg.SHUFF_BUFFER)
      dataset = dataset.batch(self._batch_size).prefetch(10)
      self._data_iter = dataset.make_initializable_iterator()
      self._x_in, self._y_labels = self._data_iter.get_next()

      device = '/device:GPU:0' if self._cfg.GPU else '/cpu:0'
      with self._graph.device(device):
        self._U = tf.get_variable(dtype=self._DTYPE, shape=[self._V, self._cfg.D], name='U')
        rep = tf.matmul(self._x_in, self._U)   # shape [batch_size, D]
        self._W = tf.get_variable(dtype=self._DTYPE,
                               shape=[self._cfg.D, self._N_HEADS*self._V],
                               name='W')
        logits = tf.matmul(rep, self._W)  # shape [batch_size, N*V]

        self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=self._y_labels, logits=logits
        ))
        optimizer = tf.train.AdamOptimizer()
        self._train_step = optimizer.minimize(self._loss)

  def _data_generator(self, corpus_path, window_sizes):
    gen = data_generator(corpus_path, window_sizes, self._xy_arrays)
    while True:
      yield gen.__next__()

  def _xy_arrays(self, window):
    """This method takes a context `window` (an iterable of characters) and return 2 arrays `X, Y` where `X` is the encoding of the middle character, and `Y` is the _context vector_."""
    midpos = int((len(window) - 1)/2)
    X = self._tokenizer.to_1hot(window[midpos]).flatten()  #length V
    Ys = []
    for p in self._cfg.WINDOW_SIZES:
      t_left = window[midpos - p]
      t_right = window[midpos + p]
      Ys.append(self._tokenizer.to_1hot(t_left).flatten())
      Ys.append(self._tokenizer.to_1hot(t_right).flatten())
    Y = np.concatenate(Ys)
    return X, Y

  def train(self):
    """Pre-defined training script"""
    if self._graph is None:
      self._create_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=self._graph, config=config) as sess:
      sess.run(tf.global_variables_initializer())

      # Do training & fine-tune
      self._train(sess, print_every=3000)
      self._train(sess, n_steps=501, batch_size=512, print_every=500)

      # Save result and close sess
      self.U_ = sess.run(self._U)  #The character embedding matrix
      self.W_ = sess.run(self._W)  #The context prediction matrix

  def _train(self, sess, n_steps=None, batch_size=None, print_every=500):
    """Train the model using sess, with optional params `n_steps` and `batch_size` to override default values."""
    assert self._graph is not None
    assert sess is not None
    # Preparations
    if n_steps is None:
      n_steps = self._cfg.TOTAL_STEPS
    if batch_size is None:
      batch_size = self._cfg.BATCH

    # Training
    with self._graph.as_default():
      sess.run(self._data_iter.initializer,
                     feed_dict={self._batch_size : batch_size})
      print("Training {} steps with batch size {}...".format(n_steps, batch_size))
      for i in range(n_steps):
        loss, _ = sess.run([self._loss, self._train_step])
        if i%print_every == 0:
          print("Step {:7d}:  loss={}".format(i, loss))
