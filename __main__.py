from __future__ import print_function
import os.path
import time, math, sys
from copy import deepcopy
from sklearn.preprocessing import normalize
import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import blogDWdata as input_data
#import network_data as input_data
#import ptb_data as input_data
import network as architecture
import Config as cfg
import Calculate_Performace as perf
from Utils import labels_to_onehot, sample
#import Model

#Code structure inspired from Stanford's cs224d assignment starter codes
#class DNN(Model):
class RNNLM_v1(object):
    def load_data(self):
        # Get the 'encoded data'
        self.data_sets =  input_data.read_data_sets(self.config.train_dir)
        debug = self.config.debug
        if debug:
	    print('##############--------- Debug mode ')
            num_debug = 1024
            self.data_sets.train._x = self.data_sets.train._x[:num_debug]
            self.data_sets.validation._x  = self.data_sets.validation._x[:num_debug]
            #self.data_sets.test_x  = self.data_sets.test_x[:num_debug]
        
        self.config.data_sets._len_vocab = self.data_sets.train.vocab.__len__()
        print('--------- Vocabulary Length: '+str(self.config.data_sets._len_vocab))

    def add_placeholders(self):
        #self.data_placeholder = tf.placeholder(tf.int32,shape=[None,self.config.num_steps],name='Input')
	#self.label_placeholder = tf.placeholder(tf.int32,shape=[None,self.config.num_steps],name='Target')
	self.data_placeholder = tf.placeholder(tf.int32,name='Input')
        self.label_placeholder = tf.placeholder(tf.int32,name='Target')
        self.keep_prob = tf.placeholder(tf.float32)
    	#self.metrics = tf.placeholder(tf.float32,shape=(len(self.config.metrics),))

    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {
            self.data_placeholder: input_batch,
            self.label_placeholder: label_batch,
        }
        return feed_dict

    def add_network(self, config):
        return architecture.Network(config)

    def add_metrics(self):
        """assign and add summary to a metric tensor"""
        for i,metric in enumerate(self.config.metrics):
            tf.scalar_summary(metric, self.metrics[i])

    def add_summaries(self,sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_train = tf.train.SummaryWriter(self.config.logs_dir+"train", sess.graph)
        self.summary_writer_val = tf.train.SummaryWriter(self.config.logs_dir+"val", sess.graph)
    
    def write_summary(self,sess,summary_writer, metric_values, step, feed_dict):
        summary = self.merged_summary
        #feed_dict[self.loss]=loss
        feed_dict[self.metrics]=metric_values
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()


    def run_epoch(self, sess, dataset, train_op=None, summary_writer=None,verbose=10):
	if not train_op :
		train_op = tf.no_op()
		keep_prob = 1
	else:
		keep_prob = self.config.architecture._dropout
        # And then after everything is built, start the training loop.
        total_loss = []
	totnxt_loss = []
	totemb_loss = []
        total_steps = sum(1 for x in dataset.next_batch(self.config.batch_size,self.config.num_steps))	
	#Sets to state to zero for a new epoch
	state = self.arch.initial_state.eval()
        for step, (input_batch, label_batch) in enumerate(
            dataset.next_batch(self.config.batch_size,self.config.num_steps)):

            feed_dict = self.create_feed_dict(input_batch, label_batch)
            feed_dict[self.keep_prob] = keep_prob
	    #Set's the initial_state temporarily to the previous final state for the session  "AWESOME" -- verified
	    #feed_dict[self.arch.initial_state] = state 
	    
	    #Writes loss summary @last step of the epoch
	    if (step+1) < total_steps:
	            _, loss_value, nxt_loss, emb_loss, state = sess.run([train_op, self.loss, self.nxt_loss,self.emb_loss, self.arch.final_state], feed_dict=feed_dict)
	    else:
	            _, loss_value, nxt_loss, emb_loss, state, summary = sess.run([train_op, self.loss, self.nxt_loss,self.emb_loss, self.arch.final_state,self.summary], feed_dict=feed_dict)
		    if summary_writer != None:
			    summary_writer.add_summary(summary,self.arch.global_step.eval(session=sess))
			    summary_writer.flush()
            total_loss.append(loss_value)
            totnxt_loss.append(nxt_loss)
            totemb_loss.append(emb_loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {} nxt = {} emb = {}'.format(step, total_steps, np.exp(np.mean(total_loss)),np.mean(totnxt_loss),np.mean(totemb_loss) ))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss)),np.mean(total_loss),np.mean(totnxt_loss),np.mean(totemb_loss)

    def fit(self, sess):
        #define parametrs for early stopping early stopping
        max_epochs = self.config.max_epochs
        patience = self.config.patience  # look as this many examples regardless
        patience_increase = self.config.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.config.improvement_threshold  # a relative improvement of this much is
                                                             # considered significant
        
        # go through this many minibatches before checking the network on the validation set
        # Here we check every epoch
        validation_loss = 1e6
        done_looping = False
        step = 1
        best_step = -1
        losses = []
	learning_rate = self.config.solver._parameters['learning_rate']
        while (step <= self.config.max_epochs) and (not done_looping):
            #print 'Epoch {}'.format(epoch)
	    #step_incr_op = tf.assign_add(self.global_step,1)
	    sess.run([self.step_incr_op])
	    epoch = self.arch.global_step.eval(session=sess)

    	    start_time = time.time()
	    tr_pp, average_loss,_,_ = self.run_epoch(sess,self.data_sets.train,train_op=self.train,summary_writer=self.summary_writer_train)
	    duration = time.time() - start_time

            if (epoch % self.config.val_epochs_freq == 0):
                val_pp,val_loss,valnxt_loss,valemb_loss= self.run_epoch(sess,self.data_sets.validation,summary_writer=self.summary_writer_val)

                print('Epoch %d: tr_loss = %.2f, val_loss = %.2f nxt = %.2f emb = %.6f || tr_pp = %.2f, val_pp = %.2f  (%.3f sec)'
                      % (epoch, average_loss, val_loss, valnxt_loss, valemb_loss,tr_pp, val_pp, duration))
                	
                # Save model only if the improvement is significant
#                if (val_loss < validation_loss * improvement_threshold) and (epoch > self.config.save_epochs_after):
                if (average_loss < validation_loss * improvement_threshold) and (epoch > self.config.save_epochs_after):
                    patience = max(patience, epoch * patience_increase)
                    validation_loss = average_loss
                    checkpoint_file = os.path.join(self.config.ckpt_dir, 'checkpoint')
                    self.saver.save(sess, checkpoint_file, global_step=epoch)
                    best_step = epoch
		    patience = epoch + max(self.config.val_epochs_freq,self.config.patience_increase)
                    #print('best step %d'%(best_step))
		
		elif average_loss > validation_loss * improvement_threshold:
		    patience = epoch - 1
		    
	    else:
		    # Print status to stdout.
		    print('Epoch %d: loss = %.2f pp = %.2f (%.3f sec)' % (epoch, average_loss, tr_pp, duration))

            if (patience <= epoch):
		#config.val_epochs_freq = 2
		learning_rate = learning_rate / 10
		self.optimizer = tf.train.AdamOptimizer(learning_rate)
		patience = epoch + max(self.config.val_epochs_freq,self.config.patience_increase)
		print('--------- Learning rate dropped to: %f'%(learning_rate))		
		if learning_rate <= 0.0000001:
			print('Stopping by patience method')
	                done_looping = True

            losses.append(average_loss) 
            step += 1

        return losses, best_step

    def get_embedding(self,sess,data):
	feed_dict = {self.data_placeholder: [data], self.keep_prob: 1, self.arch.initial_state: self.arch.initial_state.eval()}
	return sess.run(self.inputs,feed_dict=feed_dict)[0]

    def get_hidden_state(self,sess,data,eos_embed=None):
	if eos_embed is None:
		eos_embed = self.arch.initial_state.eval()
	feed_dict = {self.data_placeholder: [data], self.keep_prob: 1, self.arch.initial_state: eos_embed}
	return sess.run(self.rnn_outputs,feed_dict=feed_dict)[0]

    def generate_text(self,session, starting_text='<eos>',stop_length=100, stop_tokens=None, temp=1.0 ):
	"""Generate text from the model.
	  Args:
	    session: tf.Session() object
	    starting_text: Initial text passed to model.
	  Returns:
	    output: List of word idxs
	"""
#	state = self.arch.initial_state.eval()
	state = self.get_hidden_state(session,[self.data_sets.train.vocab.encode('<eos>')],None)
	# Imagine tokens as a batch size of one, length of len(tokens[0])
	tokens = [self.data_sets.train.vocab.encode(word) for word in starting_text.split()]
	for i in xrange(stop_length):
	    	feed = {self.data_placeholder: [tokens[-1:]],
			    self.arch.initial_state: state,
			    self.keep_prob: 1}
	    	state, y_pred, embed = session.run([self.arch.final_state, self.predictions[-1],self.inputs], feed_dict=feed)
		next_word_idx = sample(y_pred[0], temperature=temp)
		tokens.append(next_word_idx)
		if stop_tokens and self.data_sets.train.vocab.decode(tokens[-1]) in stop_tokens:
		      break
	output = [self.data_sets.train.vocab.decode(word_idx) for word_idx in tokens]
	return output

       #def generate_sentence(self,session,starting_text,temp):  
    def generate_sentence(self,session,*args, **kwargs):
	"""Convenice to generate a sentence from the model."""
	return self.generate_text(session, *args, stop_tokens=['<eos>'], **kwargs)

    def __init__(self, config):
        self.config = config
        # Generate placeholders for the images and labels.
        self.load_data()
        self.add_placeholders()
        #self.add_metrics()

        # Build model
        self.arch = self.add_network(config)
        self.inputs = self.arch.embedding(self.data_placeholder)
        self.rnn_outputs = self.arch.predict(self.inputs,self.keep_prob)
        self.outputs = self.arch.projection(self.rnn_outputs)
        # casting to handle numerical stability
        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        output = tf.reshape(tf.concat(1, self.outputs), [-1, self.config.data_sets._len_vocab])
#	output = tf.reshape(tf.concat(1, self.predictions), [-1, self.config.data_sets._len_vocab])
        self.loss,self.nxt_loss,self.emb_loss = self.arch.loss(output,self.inputs,self.label_placeholder)
	self.optimizer = self.config.solver._parameters['optimizer']
        self.train = self.arch.training(self.loss,self.optimizer)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        self.summary = tf.merge_all_summaries()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step+1)
	self.init = tf.initialize_all_variables()

########END OF CLASS MODEL#############################################################################################################

def init_Model(config):
    tf.reset_default_graph()
    with tf.variable_scope('RNNLM',reuse=None) as scope:
	 model = RNNLM_v1(config)
    
    tfconfig = tf.ConfigProto( allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        load_ckpt_dir = config.ckpt_dir
	print('--------- Loading variables from checkpoint if available')
    else:
        load_ckpt_dir = ''
	print('--------- Training from scratch')
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir,config=tfconfig)
    return model, sess

def train_DNNModel():
    print('############## Training Module ')
    config = cfg.Config()
    model,sess = init_Model(config)
    with sess:
	    model.add_summaries(sess)
	    losses, best_step = model.fit(sess)

def test_DNNModel():
    print('############## Test Module ')
    config = cfg.Config()
    model,sess = init_Model(config)    
    with sess:
        test_pp = model.run_epoch(sess,model.data_sets.validation)
        print('=-=' * 5)
        print('Test perplexity: {}'.format(test_pp))
        print('=-=' * 5)

def interactive_generate_text_DNNModel():
    print('############## Generate Text Module ')
    config = cfg.Config()
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    with sess:
	starting_text = '2'
        while starting_text:
          print(' '.join(model.generate_sentence(sess, starting_text=starting_text, temp=1.0)))
          starting_text = raw_input('> ')

def dump_generate_text_DNNModel():
    print('############## Generate sentences for all words in dictionary and Dump  ')
    config = cfg.Config()
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    num_sentences = 2
    with sess:
	ignore_list = ['0','<eos>','<unk>'] 
	keys = [int(word) for word in model.data_sets.train.vocab.word_freq.keys() if word not in ignore_list] 
	keys.sort()
	vocab_len = len(keys)
	f_id = open(os.path.join(config.dataset_name+'_data.sentences'),'w')

	for starting_text in keys:
		for n in range(num_sentences):
			  words = model.generate_sentence(sess, starting_text=str(starting_text), temp=1.0)
			  f_id.write((' '.join(words[:-1])+'\n'))

def save_Embeddings_DNNModel():
    print('############## Save Embeddings Module ')
    config = cfg.Config()
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    with sess:
        model.add_summaries(sess)
        ignore_list = ['0','<eos>','<unk>'] 
        keys = [int(word) for word in model.data_sets.train.vocab.word_freq.keys() if word not in ignore_list] 
        keys.sort()
        vocab_len = len(keys)
        enc_words = np.array([model.data_sets.train.vocab.encode(str(word)) for word in keys])
        embed = np.zeros([vocab_len,model.config.mRNN._embed_size])
        for i,word in enumerate(enc_words):
            embed[i] = model.get_embedding(sess,[word],)
        fn = os.path.join('Embeddings/'+config.dataset_name+'_data.embd')
        np.savetxt(fn,embed, delimiter=',')
	print('--------- Embeddings are saved to '+fn)

        embed = np.zeros([vocab_len,model.config.mRNN._hidden_size])
        eos_embed = model.get_hidden_state(sess,[model.data_sets.train.vocab.encode('<eos>')],None)

        for i,word in enumerate(enc_words):
            embed[i] = model.get_hidden_state(sess,[word],eos_embed)
        fn = os.path.join('Embeddings/'+config.dataset_name+'_dataH.embd')
        np.savetxt(fn,embed, delimiter=',')
#        np.savetxt(fn,normalize(embed,norm='l2',axis=1), delimiter=',')
        print('--------- Embeddings are saved to '+fn)

def visualize_Embeddings_DNNModel():
    print('############## Visualize Embeddings Module ')
    config = cfg.Config()
    tf.reset_default_graph()
    sess = tf.Session()
    fn = os.path.join('Embeddings/'+config.dataset_name+'_data.embd')
    print('--------- Embeddings are loaded from dir: '+fn)
    embed = np.loadtxt(fn,delimiter=',')
    embed_var = tf.Variable(embed,name='embed_var')
    init = tf.initialize_all_variables()
    sess.run(init)

    checkpoint_file = os.path.join(config.logs_dir, 'Embedding')
    saver = tf.train.Saver({"embedding": embed_var},write_version=tf.train.SaverDef.V2)
    fn = 'Embeddings/embedding_ckpt'
    saver.save(sess,fn, global_step=1)
    print('--------- To Visualize Embeddings load tf:0.12v tensorboard in directory: '+fn)

if __name__ == "__main__": 	
	with tf.device('/gpu:1'):
	    train_DNNModel() 
	    #test_DNNModel() 
	    #interactive_generate_text_DNNModel()
	    #dump_generate_text_DNNModel()
	    save_Embeddings_DNNModel()
	    #visualize_Embeddings_DNNModel()
