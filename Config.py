import tensorflow as tf
import sys

class Config(object):

    def __init__(self):
        pass
    codebase_root_path = '/home/priyesh/Desktop/Codes/'
    sys.path.insert(0, codebase_root_path)

    ####  Directory paths ####
    #Folder name and project name is the same
    project_name = 'LM_AE_128_128Tr_E'
    dataset_name = 'blogDWdata'
    #dataset_name = 'ptb'	
   
    train_dir = codebase_root_path+project_name+'/'+dataset_name+'/'

    #Logs and checkpoints to be stored in the code directory
    project_prefix_path = codebase_root_path+project_name+'/'
    logs_dir = project_prefix_path+'Logs/'
    #ckpt_dir = codebase_root_path+'LM-v1.3'+'/'+'Checkpoints/'
    ckpt_dir = './Checkpoints/'

    #Retrain
    retrain = True

    #Debug with small dataset
    debug = False

    # Batch size
    batch_size = 64
    num_steps = 7
    #Number of steps to run trainer
    max_epochs = 100
    #Validation frequence
    val_epochs_freq = 1
    #Model save frequency
    save_epochs_after= 0

    #Other hyperparameters
    #thresholding constant
    th=0.4

    #earlystopping hyperparametrs
    patience = max_epochs # look as this many epochs regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.9999  # a relative improvement of this much is considered significant

    metrics = ['coverage','average_precision','ranking_loss','micro_f1','macro_f1','micro_precision',
               'macro_precision','micro_recall','macro_recall','p@1','p@3','p@5']
    
    class Solver(object):
        def __init__(self):
            self._parameters = {}
            #Initial learning rate
            self._parameters['learning_rate'] = 0.001

            #optimizer
            self._parameters['optimizer'] = tf.train.AdamOptimizer(self._parameters['learning_rate'])

    class Architecture(object):
        def __init__(self):
            self._parameters = {}
            #Number of layer - excluding the input & output layers
            self._parameters['num_layers'] = 2
            #Mention the number of layers
            self._parameters['layers'] = [100,250]
            #dropout
            self._dropout = 0.9

    class Data_sets(object):
        def __init__(self):
            self._len_vocab = 0
    

    class RNNArchitecture(object):
        def __init__(self):
            #self._num_steps = 10 # Problem with reusing variable
            self._embed_size = 128
            self._hidden_size = 128
            self._dropout = 1

    solver = Solver()
    architecture = Architecture()
    data_sets = Data_sets()
    mRNN = RNNArchitecture()

    # TO DO
    #http://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
