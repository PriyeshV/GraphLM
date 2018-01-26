from collections import defaultdict
import numpy as np

def data_iterator(orig_X, orig_y=None, batch_size=32, shuffle=False):
	# Optionally shuffle the data before training
	# pulled from cs224d assignments and modified
	if shuffle:
		indices = np.random.permutation(len(orig_X))
		data_X = orig_X[indices]
		data_y = orig_y[indices] if np.any(orig_y) else None
	else:
		data_X = orig_X
		data_y = orig_y
	###
	total_processed_examples = 0
	total_steps = int(len(data_X) // batch_size)
	for step in xrange(total_steps):
		# Create the batch by selecting up to batch_size elements
		batch_start = step * batch_size
		x = data_X[batch_start:batch_start + batch_size]
		y = data_y[batch_start:batch_start + batch_size]
		yield x, y
		total_processed_examples += len(x)
	# Sanity check to make sure we iterated over all the dataset as intended
	assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)

def labels_to_onehot(data_y,num_instances,num_labels):
	#assuming the labes start with 0
	y = np.zeros((num_instances, num_labels), dtype=np.int32)
	y[np.arange(num_instances, dtype=np.int32), data_y] = 1
	return y

def ptb_iterator(raw_data, batch_size, num_steps,shuffle):
    # Pulled from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py#L82 and modified
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = int(data_len // batch_size)
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    if shuffle:
	indices = np.random.permutation(len(data))
	data = data[indices]
    epoch_size = (batch_len - 1) // num_steps
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)

###########################################

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def calculate_perplexity(log_probs):
  # https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf
  perp = 0
  for p in log_probs:
    perp += -p
  return np.exp(perp / len(log_probs))
