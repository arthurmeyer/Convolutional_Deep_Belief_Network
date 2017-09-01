import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data
import cdbn_backup as cdbn


""" --------------------------------------------
    ------------------- DATA -------------------
    -------------------------------------------- """

class MNIST_HANDLER(object):
  
  def __init__(self, data):
    self.num_training_example = data.train.num_examples
    self.num_test_example     = data.test.num_examples
    self.training_data , self.training_labels = data.train.next_batch(self.num_training_example)
    self.test_data , self.test_labels        = data.test.next_batch(self.num_test_example)
    self.whiten               = False
    self.training_index       = -20
    self.test_index           = -20
    
    
  def do_whiten(self):
    self.whiten         = True
    data_to_be_whitened = np.copy(self.training_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_training_example
    mean                = np.tile(mean,self.num_training_example)
    mean                = np.reshape(mean,(self.num_training_example,784))
    centered_data       = data_to_be_whitened - mean                
    covariance          = np.dot(centered_data.T,centered_data)/self.num_training_example
    U,S,V               = np.linalg.svd(covariance)
    epsilon = 1e-5
    lambda_square       = np.diag(1./np.sqrt(S+epsilon))
    self.whitening_mat  = np.dot(np.dot(U, lambda_square), V)    
    self.whitened_training_data  = np.dot(centered_data,self.whitening_mat)
    
    data_to_be_whitened = np.copy(self.test_data)
    mean                = np.sum(data_to_be_whitened, axis = 0)/self.num_test_example
    mean                = np.tile(mean,self.num_test_example)
    mean                = np.reshape(mean,(self.num_test_example,784))
    centered_data       = data_to_be_whitened - mean  
    self.whitened_test_data  = np.dot(centered_data,self.whitening_mat)

    
  def next_batch(self, batch_size, type = 'train'):
    if type == 'train':
      if self.whiten:
        operand = self.whitened_training_data
      else:
        operand = self.training_data
      operand_bis = self.training_labels
      self.training_index = (batch_size + self.training_index) % self.num_training_example
      index = self.training_index
      number = self.num_training_example
    elif type == 'test':
      if self.whiten:
        operand = self.whitened_test_data
      else:
        operand = self.test_data
      operand_bis = self.test_labels
      self.test_index = (batch_size + self.test_index) % self.num_test_example
      index = self.test_index
      number = self.num_test_example

    if index + batch_size > number:
      part1 = operand[index:,:]
      part2 = operand[:(index + batch_size)% number,:]
      result = np.concatenate([part1, part2])
      part1 = operand_bis[index:,:]
      part2 = operand_bis[:(index + batch_size)% number,:]
      result_bis = np.concatenate([part1, part2])
    else:
      result = operand[index:index + batch_size,:]
      result_bis = operand_bis[index:index + batch_size,:]
    return result, result_bis
        

mnist_dataset = MNIST_HANDLER(input_data.read_data_sets('data', one_hot=True))
#mnist_dataset.do_whiten()
sess = tf.Session()




""" ---------------------------------------------
    ------------------- MODEL -------------------
    --------------------------------------------- """

my_cdbn = cdbn.CDBN('mnist_cdbn', 20, '/home/arthur/pedestrian_detection/log', mnist_dataset, sess, verbosity = 2)

my_cdbn.add_layer('layer_1', fully_connected = False, v_height = 28, v_width = 28, v_channels = 1, f_height = 11, f_width = 11, f_number = 40, 
               init_biases_H = -3, init_biases_V = 0.01, init_weight_stddev = 0.01, 
               gaussian_unit = True, gaussian_variance = 0.2, 
               prob_maxpooling = True, padding = True, 
               learning_rate = 0.00005, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 2.0, sparsity_target = 0.003, sparsity_coef = 0.1)

my_cdbn.add_layer('layer_2', fully_connected = False, v_height = 14, v_width = 14, v_channels = 40, f_height = 7, f_width = 7, f_number = 40, 
               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = True, padding = True, 
               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)

my_cdbn.add_layer('layer_3', fully_connected = True, v_height = 1, v_width = 1, v_channels = 40*7*7, f_height = 1, f_width = 1, f_number = 200, 
               init_biases_H = -3, init_biases_V = 0.025, init_weight_stddev = 0.025, 
               gaussian_unit = False, gaussian_variance = 0.2, 
               prob_maxpooling = False, padding = False, 
               learning_rate = 0.0025, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1)

my_cdbn.add_softmax_layer(10, 0.1)

my_cdbn.lock_cdbn()




""" ---------------------------------------------
    ------------------ TRAINING -----------------
    --------------------------------------------- """
my_cdbn.manage_layers(['layer_1','layer_2','layer_3'],[],[10000,10000,10000], [1,1,1], 20000, restore_softmax = False, fine_tune = True)
my_cdbn.do_eval()