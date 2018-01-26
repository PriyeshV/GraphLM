import tensorflow as tf
import numpy as np

from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn import metrics
from collections import Counter
import math
from tensorflow.python.ops.seq2seq import sequence_loss
#from Architectures import Architecture

#class Network(Architecture):
class Network(object):
    
    def __init__(self,config):
        self.config = config
        self.initial_state = tf.zeros([self.config.batch_size, self.config.mRNN._hidden_size])
	self.global_step = tf.Variable(0,name="global_step",trainable=False) #Epoch

    def weight_variable(self,shape):
      initial = tf.truncated_normal(shape,stddev=1.0 / shape[0])
      return tf.Variable(initial)

    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            tf.histogram_summary(name, var)

    def embedding(self, inputs):
        """Add embedding layer.
        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
	self.embedding = tf.get_variable(
	  'Embedding',
	  [self.config.data_sets._len_vocab, self.config.mRNN._embed_size], trainable=True)
        inputs = tf.nn.embedding_lookup(self.embedding, inputs)
        inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]
        return inputs

    def projection(self, rnn_outputs):
        """Adds a projection layer.

        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.

        Hint: Here are the dimensions of the variables you will need to
              create 
              
              U:   (hidden_size, len(vocab))
              b_2: (len(vocab),)

        Args:
          rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab)
        """
#        with tf.variable_scope('Projection'):
#            U = tf.get_variable(
#              'Matrix', [self.config.mRNN._hidden_size, self.config.data_sets._len_vocab])
#            proj_b = tf.get_variable('Bias', [self.config.data_sets._len_vocab])
#            outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]
	with tf.variable_scope('RNN',reuse=True):
	    h_outputs = [tf.matmul(o,tf.transpose(self.RNN_I)) for o in rnn_outputs]
	outputs = [tf.matmul(o,tf.transpose(self.embedding)) for o in h_outputs]
        return outputs


    def predict(self,inputs,keep_prob):
        """Build the model up to where it may be used for inference.
        """
        hidden_size = self.config.mRNN._hidden_size
        batch_size = self.config.batch_size
        embed_size = self.config.mRNN._embed_size

        if keep_prob == None:
            keep_prob = 1

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x,keep_prob) for x in inputs]
                
        with tf.variable_scope('RNN') as scope:
	    state = self.initial_state
            RNN_H = tf.get_variable('HMatrix',[hidden_size,hidden_size])
            self.RNN_I = tf.get_variable('IMatrix', [embed_size,hidden_size])
            self.RNN_b = tf.get_variable('B',[hidden_size])
        
        with tf.variable_scope('RNN',reuse=True):
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                RNN_H = tf.get_variable('HMatrix',[hidden_size,hidden_size])
                self.RNN_I = tf.get_variable('IMatrix', [embed_size,hidden_size])
                self.RNN_b = tf.get_variable('B',[hidden_size])
                state = tf.nn.sigmoid(tf.matmul(state,RNN_H) + tf.matmul(current_input,self.RNN_I) + self.RNN_b)
                rnn_outputs.append(state)
		#How to pass state info for subsequent sentences
            self.final_state = rnn_outputs[-1]
    
        with tf.variable_scope('RNNDropout'):
            rnn_outputs = [tf.nn.dropout(x,keep_prob) for x in rnn_outputs]

        return rnn_outputs

    def loss(self, predictions, rnn_outputs, labels):
        """Calculates the loss from the predictions (logits?) and the labels.
        """
        all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
        cross_entropy = sequence_loss([predictions],
                                      [tf.reshape(labels, [-1])], all_ones, self.config.data_sets._len_vocab)
        tf.add_to_collection('total_loss', cross_entropy)

	with tf.variable_scope('Embedding_similarity'):
		    num_steps = 7
		    if self.config.num_steps != 1:
			    h1 = [o1*o2 for o1,o2 in zip(rnn_outputs[:-2],rnn_outputs[1:-1])]
			    h2 = [o1*o2 for o1,o2 in zip(rnn_outputs[1:-1],rnn_outputs[2:])]
			    emb_sim = tf.reduce_mean(tf.exp(tf.square(tf.sub(h1,h2))))
		    else:
			    h1 = [rnn_outputs[0]*rnn_outputs[0] for i in range(num_steps-2)]
			    h2 = [rnn_outputs[0]*rnn_outputs[0] for i in range(num_steps-2)]
			    emb_sim = tf.reduce_mean(tf.exp(tf.square(tf.sub(h1,h2))))	

	            tf.add_to_collection('total_loss', emb_sim)


        loss = tf.add_n(tf.get_collection('total_loss'))
        return loss, cross_entropy, emb_sim

    def training(self, loss, optimizer):
      """Sets up the training Ops.
      Creates a summarizer to track the loss over time in TensorBoard.
      Creates an optimizer and applies the gradients to all trainable variables.
      The Op returned by this function is what must be passed to the
      `sess.run()` call to cause the model to train.
      Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
      Returns:
        train_op: The Op for training.
      """
      # Add a scalar summary for the snapshot loss.
      tf.scalar_summary('loss', loss)
      # Create a variable to track the global step. - iterations
      #global_step = tf.Variable(0, name='global_step', trainable=False)
      #train_op = optimizer.minimize(loss, global_step=global_step)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(loss)
      return train_op

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
    """Generate text from the model.

    Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
    Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
    Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

    Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
    Returns:
    output: List of word idxs
    """
    state = model.initial_state.eval()
    # Imagine tokens as a batch size of one, length of len(tokens[0])
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    for i in xrange(stop_length):
        feed = {model.input_placeholder: [tokens[-1:]],
                model.initial_state: state,
                model.dropout_placeholder: 1}
        state, y_pred = session.run(
            [model.final_state, model.predictions[-1]], feed_dict=feed)
        next_word_idx = sample(y_pred[0], temperature=temp)
        tokens.append(next_word_idx)
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
          break
    output = [model.vocab.decode(word_idx) for word_idx in tokens]
    return output

def generate_sentence(session, model, config, *args, **kwargs):
    """Convenice to generate a sentence from the model."""
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)
