from __future__ import division
import tensorflow as tf
import numpy as np
import time
import os
import crbm_backup as crbm


class CDBN(object):
  """CONVOLUTIONAL DEEP BELIEF NETWORK"""
  
  def __init__(self, name, batch_size, path, data, session, verbosity = 2):
    """INTENT : Initialization of a Convolutional Deep Belief Network
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    name         :        name of the CDBN
    batch_size   :        batch size to work with  
    path         :        where to save and restore parameter of trained layer
    train_data   :        data to use the CDBN for training
    test_data    :        data to use the CDBN for testing
    session      :        tensorflow session (context) to use this CDBN in
    verbosity    :        verbosity of the training  (0 is low  1 is medium and 2 is high)
    ------------------------------------------------------------------------------------------------------------------------------------------
    ATTRIBUTS :
    number_layer             :        number of layer (is updated everytime add_layer() method is called
    layer_name_to_object     :        link between layer name and their corresponding crbm object
    layer_level_to_name      :        link between layer level and it name
    layer_name_to_level      :        link between layer name and it level
    input                    :        shape of the visible layer of the first layer ie where the data is to be clamped to
    fully_connected_layer    :        where the first fully connected layer occur
    locked                   :        if the CDBN model is completed ie all layer have been added
    softmax_layer            :        if the model has a softmax layer on top"""
    
    self.name                       =     name
    self.batch_size                 =     batch_size 
    self.path                       =     path + "/" + name
    tf.gfile.MakeDirs(self.path) 
    self.data                       =     data
    self.session                    =     session
    self.verbosity                  =     verbosity
    self.number_layer               =     0
    self.layer_name_to_object       =     {}
    self.layer_level_to_name        =     {}
    self.layer_name_to_level        =     {}
    self.input                      =     None
    self.fully_connected_layer      =     None
    self.locked                     =     False
    self.softmax_layer              =     False 
        
        
        

        
  def add_layer(self, name, fully_connected = True, v_height = 1, v_width = 1, v_channels = 784, f_height = 1, f_width = 1, f_number = 400, 
               init_biases_H = -3, init_biases_V = 0.01, init_weight_stddev = 0.01, 
               gaussian_unit = True, gaussian_variance = 0.2, 
               prob_maxpooling = False, padding = False, 
               learning_rate = 0.0001, learning_rate_decay = 0.5, momentum = 0.9, decay_step = 50000,  
               weight_decay = 0.1, sparsity_target = 0.1, sparsity_coef = 0.1):
    """INTENT : Add a layer to the CDBN (on the top)
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS : (same as for CRBM)
    name                  :         name of the RBM
    fully_connected       :         specify if the RBM is fully connected (True) or convolutional (False)     |   if True then obviously all height and width are 1
    v_height              :         height of the visible layer (input)
    v_width               :         width of the visible layer (input)
    v_channels            :         numbers of channels of the visible layer (input)
    f_height              :         height of the filter to apply to the visible layer 
    f_width               :         width of the filter to apply to the visible layer 
    f_number              :         number of filters to apply to the visible layer
    init_biases_H         :         initialization value for the bias of the hidden layer
    init_biases_V         :         initialization value for the bias of the visible layer
    init_weight_stddev    :         initialization value of the standard deviation for the kernel
    gaussian_unit         :         True if using gaussian unit for the visible layer, false if using binary unit
    gaussian_variance     :         Value of the variance of the gaussian distribution of the visible layer (only for gaussian visible unit)
    prob_maxpooling       :         True if the CRBM also include a probabilistic max pooling layer on top of the hidden layer (only for convolutional RBM)
    padding               :         True if the visible and hidden layer have same dimension (only for convolutional RBM)
    learning_rate         :     learning rate for gradient update    
    learning_rate_decay   :     value of the exponential decay
    momentum              :     coefficient of the momemtum in the gradient descent
    decay_step            :     number of step before applying gradient decay
    weight_decay          :     coefficient of the weight l2 norm regularization
    sparsity_target       :     probability target of the activation of the hidden units
    sparsity_coef         :     coefficient of the sparsity regularization term
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : Dynamically update CDBN global view of the model"""
    
    try:
      if self.locked:
        raise ValueError('Trying to add layer ' + name + ' to CDBN ' + self.name + ' which has already been locked') 
      if name == 'softmax_layer':
        raise ValueError('Trying to add layer ' + name + ' to CDBN ' + self.name + ' but this name is protected') 
      if name in self.layer_name_to_object:
        raise ValueError('Trying to add layer ' + name + ' to CDBN ' + self.name + ' but this name is already use') 
      else:
        self.layer_level_to_name[self.number_layer]  =  name   
        self.layer_name_to_level[name]               =  self.number_layer
        if self.input is None:
          self.input = (self.batch_size,v_height,v_width,v_channels)
        elif not fully_connected:
          ret_out    = self.layer_name_to_object[self.layer_level_to_name[self.number_layer - 1]]
          if not (v_height   == ret_out.hidden_height / (ret_out.prob_maxpooling + 1)):
            raise ValueError('Trying to add layer ' + name + ' to CDBN ' + self.name + ' which height of visible layer does not match height of output of previous layer') 
          if not (v_width    == ret_out.hidden_width  / (ret_out.prob_maxpooling + 1)):
            raise ValueError('Trying to add layer ' + name + ' to CDBN ' + self.name + ' which width of visible layer does not match width of output of previous layer') 
          if not (v_channels == ret_out.filter_number):
            raise ValueError('Trying to add layer ' + name + ' to CDBN ' + self.name + ' which number of channels of visible layer does not match number of channels of output of previous layer')  
        if fully_connected and self.fully_connected_layer is None:
          self.fully_connected_layer = self.number_layer
        self.layer_name_to_object[name] = crbm.CRBM(name, fully_connected, v_height, v_width, v_channels, f_height, f_width, f_number, 
                                                init_biases_H, init_biases_V, init_weight_stddev, 
                                                gaussian_unit, gaussian_variance, 
                                                prob_maxpooling, padding,
                                                self.batch_size, learning_rate, learning_rate_decay, momentum, decay_step,  
                                                weight_decay, sparsity_target, sparsity_coef)
        self.number_layer = self.number_layer + 1
        'Where to save and restore parameter of this layer'
        tf.gfile.MakeDirs(self.path + "/" + name) 
        
        if self.verbosity > 0:
          print('--------------------------')
        if fully_connected:
          message = 'Successfully adding fully connected layer ' + name + ' to CDBN ' + self.name
          if self.verbosity > 0:
            message += ' with has ' + str(v_channels) + ' visible units and ' + str(f_number) + ' hidden units '
        else:
          message = 'Successfully adding convolutional layer ' + name + ' to CDBN ' + self.name
          if self.verbosity > 0:
            message += ' with configuration of visible is ('+str(v_height)+','+str(v_width)+','+str(v_channels)+') and filters is ('+str(f_height)+','+str(f_width)+','+str(f_number)+')'
          if self.verbosity > 1 and prob_maxpooling:
            message += '\nProbabilistic max pooling ON'
          if self.verbosity > 1 and  padding:
            message += '\nPadding ON'
        if self.verbosity > 1 and gaussian_unit:
            message += '\nGaussian unit ON'
        print(message)  
          
    except ValueError as error:
      self._print_error_message(error)
      
        
        

        
  def add_softmax_layer(self, output_classes, learning_rate, fine_tune = False):
    """INTENT : add a softmax layer on top of the CDBN
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS : 
    output_classes         :    number of class for the softmax output""" 
    
    try:
      if self.locked:
        raise ValueError('Trying to add softmax layer to the CDBN ' + self.name + ' which has already been locked')
      if self.softmax_layer:
        raise ValueError('Trying to add softmax layer to the CDBN ' + self.name + ' which has already one')
      else:
        self.soft_step = 0
        self.softmax_layer  = True
        self.output_classes = output_classes
        ret_out = self.layer_name_to_object[self.layer_level_to_name[self.number_layer-1]]
        self.output = ret_out.hidden_height / (ret_out.prob_maxpooling + 1) * ret_out.hidden_width / (ret_out.prob_maxpooling + 1)  * ret_out.filter_number
        with tf.variable_scope('softmax_layer_cdbn'):
          with tf.device('/cpu:0'):
            self.W            = tf.get_variable('weights_softmax', (self.output, output_classes), initializer=tf.truncated_normal_initializer(stddev=1/self.output, dtype=tf.float32), dtype=tf.float32)
            self.b            = tf.get_variable('bias_softmax', (output_classes), initializer=tf.constant_initializer(0), dtype=tf.float32)
        tf.gfile.MakeDirs(self.path + "/" + 'softmax_layer') 
       
        if self.verbosity > 0:
          print('--------------------------')
        print('Successfully added softmax layer to the CDBN ' + self.name)
        
        lr = tf.train.exponential_decay(learning_rate,self.soft_step,35000,0.1,staircase=True)
        self.softmax_trainer = tf.train.MomentumOptimizer(lr,0.9)
        self.input_placeholder  = tf.placeholder(tf.float32, shape=self.input)
        eval = tf.reshape(self._get_input_level(self.number_layer,self.input_placeholder), [self.batch_size, -1])
        y = tf.nn.softmax(tf.matmul(eval, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None,self.output_classes])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))
        self.cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if fine_tune:
          self.train_step = self.softmax_trainer.minimize(cross_entropy)
        else:
          (ret_w_0 , ret_w_1), ret_b = self.softmax_trainer.compute_gradients(cross_entropy, var_list=[self.W,self.b])
          self.train_step = self.softmax_trainer.apply_gradients([(ret_w_0 , ret_w_1), ret_b])
          self.control =  tf.reduce_mean(tf.abs(tf.div(tf.mul(ret_w_0,learning_rate),ret_w_1))) 
    except ValueError as error:
      self._print_error_message(error)
      
        
        
        
        
  def lock_cdbn(self):
    """INTENT : lock the cdbn model""" 
    
    try:
      if self.locked:
        raise ValueError('Trying to lock CDBN ' + self.name + ' which has already been locked') 
      else:
        if not self.softmax_layer:
          ret_out = self.layer_name_to_object[self.layer_level_to_name[self.number_layer-1]]
          self.output = ret_out.hidden_height / (ret_out.prob_maxpooling + 1) * ret_out.hidden_width / (ret_out.prob_maxpooling + 1)  * ret_out.filter_number
        self.locked = True
        
        if self.verbosity > 0:
          print('--------------------------')
        print('Successfully locked the CDBN ' + self.name)
          
    except ValueError as error:
      self._print_error_message(error)
      
      
      
      
      
  def manage_layers(self, layers_to_pretrain, layers_to_restore, step_for_pretraining, n_for_pretraining, step_softmax = 0, restore_softmax = False, fine_tune = False, threaded_input = False, learning_rate = 0.5):
    """INTENT : manage the initialization / restoration of the different layers of the CDBN
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    layers_to_pretrain             :         layers to be initialized from scratch and pretrained (names list)
    layers_to_restore              :         layers to be restored (names list)
    step_for_pretraining           :         step of training for layers to be pretrained
    n_for_pretraining              :         length of the gibbs chain for pretraining
    step_softmax                   :         step for training softmax layer
    is_softmax                     :         is there a softmax layer
    restore_softmax                :         should it be restored (True) or trained from scratch (False)""" 
    
    try:
      if not self.locked:
        raise ValueError('Trying to initialize layers of CDBN ' + self.name + ' which has not been locked') 
      if len(layers_to_pretrain) != 0 and ((len(layers_to_pretrain) != len(step_for_pretraining)) or (len(layers_to_pretrain) != len(n_for_pretraining))):
        raise ValueError('Parameter given for the layer to be pretrained are not complete (ie 3rd and 4th argument should be list which length match one of the 1st arg)') 
      else:
        self.session.run(tf.initialize_all_variables())
        for layer in layers_to_pretrain:
          self._init_layer(layer, from_scratch = True)
          if self.verbosity > 0:
            print('--------------------------')
          print('Successfully initialized the layer ' + layer + ' of CDBN ' + self.name)
          
        for layer in layers_to_restore:
          self._init_layer(layer, from_scratch = False)
          
        if self.softmax_layer and not restore_softmax: 
          self.session.run(tf.initialize_variables([self.W, self.b]))
          if self.verbosity > 0:
            print('--------------------------')
          print('Successfully initialized the softmax layer of CDBN ' + self.name)
        
        for layer in layers_to_restore:
          self._restore_layer(layer)
          if self.verbosity > 0:
            print('--------------------------')
          print('Successfully restored the layer ' + layer + ' of CDBN ' + self.name)
            
        for i in range(len(layers_to_pretrain)):
          self._pretrain_layer(layers_to_pretrain[i], step_for_pretraining[i], n_for_pretraining[i])
          
        if self.softmax_layer and restore_softmax: 
          self._restore_layer('softmax_layer')
          if self.verbosity > 0:
            print('--------------------------')
          print('Successfully restored the softmax layer of CDBN ' + self.name)
         
        if self.softmax_layer and not restore_softmax: 
          self._do_softmax_training(step_softmax, fine_tune, learning_rate)
          
        for i in range(len(layers_to_pretrain)):
          self._save_layer(layers_to_pretrain[i], step_for_pretraining[i])
          if self.verbosity > 0:
            print('--------------------------')
          print('Successfully saved the layer ' + layers_to_pretrain[i] + ' of CDBN ' + self.name)
          
        if self.softmax_layer and not restore_softmax: 
          self._save_layer('softmax_layer', step_softmax)
          if self.verbosity > 0:
            print('--------------------------')
          print('Successfully saved the softmax layer of CDBN ' + self.name)
          
    except ValueError as error:
      self._print_error_message(error)
      
    
    
    
    
  def do_eval(self, f1 = False):
    """INTENT : Evaluate the CDBN as a classifier"""
    
    input_placeholder  = tf.placeholder(tf.float32, shape=self.input)
    
    eval = tf.reshape(self._get_input_level(self.number_layer,input_placeholder), [self.batch_size, -1])
    y = tf.nn.softmax(tf.matmul(eval, self.W) + self.b)
    y_ = tf.placeholder(tf.float32, [None,self.output_classes])
    
    if f1:
      predicted_class = tf.argmax(y,1)
      real_class = tf.argmax(y_,1)
      zeros = tf.zeros_like(predicted_class)
      ones  = tf.ones_like(predicted_class)
      
      true_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, ones),tf.equal(real_class, ones)), tf.float32))
      tp_count = 0
      false_positive = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, ones),tf.equal(real_class, zeros)), tf.float32))
      fp_count = 0
      true_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, zeros),tf.equal(real_class, zeros)), tf.float32))
      tn_count = 0
      false_negative = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predicted_class, zeros),tf.equal(real_class, ones)), tf.float32))
      fn_count = 0
    else:
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      correct_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
      true_count = 0
      
    num_examples = self.data.num_test_example
    steps_per_epoch = num_examples // self.batch_size
    
    for step in range(steps_per_epoch):
      images_feed, labels_feed = self.data.next_batch(self.batch_size, 'test')
      visible                  = np.reshape(images_feed, self.input)
      if f1:
        a,b,c,d = self.session.run([true_positive,false_positive,true_negative,false_negative], feed_dict={input_placeholder: visible, y_: labels_feed})
        tp_count += a
        fp_count += b
        tn_count += c
        fn_count += d
      else:
        true_count += self.session.run(correct_count, feed_dict={input_placeholder: visible, y_: labels_feed})
        
    if self.verbosity > 0:
      print('--------------------------')
    if f1:
      precision = tp_count / (tp_count+fp_count)
      recall = tp_count / (tp_count+fn_count)
      f1_score = 2*precision*recall/(precision+recall)
      overall_precision = (tp_count + tn_count) / (fn_count+ fp_count + tp_count +tn_count)
      print('Successfully evaluated the CDBN : \n Precision is %0.02f percent \n Recall is %0.02f percent \n F1 score is %0.02f\n tp: %d ---  fp: %d ---  tn: %d ---  fn: %d\n Overall precision is %0.02f percent' %(precision*100, recall*100, f1_score, tp_count, fp_count, tn_count, fn_count, overall_precision * 100))
    else:
      precision = true_count / num_examples
      print('Successfully evaluated the CDBN : \n %d examples are correctly classified out of %d total examples\n Precision is %0.02f percent' %(true_count, num_examples, precision*100))
    
        
        

  def _pretrain_layer(self, rbm_layer_name, number_step, n = 1):
    """INTENT : Pretrain the given layer
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    rbm_layer_name         :        name of CRBM layer that we want to do one step of pretaining
    number_step            :        number of step to use for training
    n                      :        length of gibbs chain to use"""
    
    start = time.time()  
    if self.verbosity > 0:
      start_t         = time.time()
      average_cost    = 0
      print('--------------------------')  
    if self.verbosity == 2:
      average_control = 0
    print('Starting training the layer ' + rbm_layer_name + ' of CDBN ' + self.name)
    if self.verbosity > 0:
      print('--------------------------')
  
    layer_input         = self.layer_name_to_object[self.layer_level_to_name[0]]
    input_placeholder   = tf.placeholder(tf.float32, shape=self.input)
    step_placeholder    = tf.placeholder(tf.int32, shape=(1))  
    input               = self._get_input_level(self.layer_name_to_level[rbm_layer_name], input_placeholder)
    a,b,c,error,control = self._one_step_pretraining(rbm_layer_name, input , n, step_placeholder)
    
    for i in range(1,number_step):
      if self.verbosity > 0:
        start_time = time.time()
      input_images, _     = self.data.next_batch(self.batch_size, 'train')
      visible             = np.reshape(input_images, self.input)
      _,_,_,err,con       = self.session.run([a,b,c,error,control], feed_dict={input_placeholder: visible, step_placeholder : np.array([i])})

      if self.verbosity > 0:
        average_cost    = average_cost + err
        duration = time.time() - start_time
      if self.verbosity == 2:
        average_control = average_control + con
        
      if self.verbosity == 1 and i % 500 == 0 and not (i % 1000 == 0):
        print('Step %d: reconstruction error = %.05f (%.3f sec)  -----  Estimated remaining time is %.0f sec' % (i, average_cost/500, duration, (number_step-i)*(time.time() - start_t)/500))
      elif self.verbosity == 1 and i % 1000 == 0:
        print('Step %d: reconstruction error = %.05f (%.3f sec)  -----  Estimated remaining time is %.0f sec' % (i, average_cost/1000, duration, (number_step-i)*(time.time() - start_t)/1000))
        average_cost    = 0
        start_t = time.time()      
        
      if self.verbosity == 2 and i % 100 == 0 and not (i % 1000 == 0):
        print('Step %d: reconstruction error = %.05f (%.3f sec) and weight upgrade to weight ratio is %.2f percent  -----  Estimated remaining time is %.0f sec' % (i, average_cost/(i % 1000), duration, average_control/(i % 1000)*1,(number_step-i)*(time.time() - start_t)/(i % 1000)))
      elif self.verbosity == 2 and i % 1000 == 0:
        print('Step %d: reconstruction error = %.05f (%.3f sec) and weight upgrade to weight ratio is %.2f percent  -----  Estimated remaining time is %.0f sec' % (i, average_cost/1000, duration, average_control/1000*1,(number_step-i)*(time.time() - start_t)/1000))
        average_cost    = 0
        average_control = 0
        start_t = time.time()  

    if self.verbosity > 0:
      print('--------------------------')       
    message = 'Successfully trained the layer ' + rbm_layer_name + ' of CDBN ' + self.name + ' in %.0f sec'
    print(message % (time.time() - start))
      
        
        
      
      
  def _do_softmax_training(self, step = 2000, fine_tune = False, learning_rate = 0.5):
    """INTENT : Train the softmax output layer of our CDBN
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    step         :        number of steps for training
    save_softmax :        whether softmax layer should be saved or not"""

    if self.verbosity > 0:
      print('--------------------------')
    print('Starting training the softmax layer of CDBN ' + self.name)
    if self.verbosity > 0:
      print('--------------------------')
    start = time.time()

    if self.verbosity == 2:
      average_cost = 0
      average_control = 0
    for i in range(1,step):
      self.soft_step += 1
      images_feed, labels_feed = self.data.next_batch(self.batch_size, 'train')
      visible                  = np.reshape(images_feed, self.input)
      if fine_tune:
        _, a = self.session.run([self.train_step,self.cross_entropy_mean], feed_dict={self.input_placeholder: visible, self.y_: labels_feed})
      else:
        _, a, b = self.session.run([self.train_step,self.cross_entropy_mean, self.control], feed_dict={self.input_placeholder: visible, self.y_: labels_feed})
        average_control += b
      average_cost += a
      if self.verbosity > 0 and i % 250 == 0:
        print('Step %d: cost is %.3f----- control value (gradient rate) : %.3f percent --- Estimated remaining time is %.0f sec' % (i, average_cost/250, average_control/250*100, (step-i)*(time.time() - start)/i))
        average_cost = 0
        average_control = 0
    if self.verbosity > 0:
      print('--------------------------')
    print('Successfully trained the softmax layer in %.0f sec' % (time.time()-start))
      
        
      
        
        
  def _save_layer(self, rbm_layer_name, step):
    """INTENT : Save given layer
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    rbm_layer_name         :        name of CRBM layer that we want to save
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : if rbm_layer_name is softmax_layer then save softmax parameter"""
      
    checkpoint_path = os.path.join(self.path + "/" + rbm_layer_name, 'model.ckpt') 
    if rbm_layer_name == 'softmax_layer':
      saver = tf.train.Saver([self.W, self.b])
      saver.save(self.session, checkpoint_path, global_step=step)
    else:
      self.layer_name_to_object[rbm_layer_name].save_parameter(checkpoint_path, self.session, step)
      
        
        

        
  def _restore_layer(self, rbm_layer_name):
    """INTENT : Restore given layer
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    rbm_layer_name         :        name of CRBM layer that we want to restore
    ------------------------------------------------------------------------------------------------------------------------------------------
    REMARK : if rbm_layer_name is softmax_layer then restore softmax parameter"""
      
    ckpt = tf.train.get_checkpoint_state(self.path + "/" + rbm_layer_name)
    if rbm_layer_name == 'softmax_layer':
      saver = tf.train.Saver([self.W, self.b])
      return saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      return self.layer_name_to_object[rbm_layer_name].load_parameter(ckpt.model_checkpoint_path, self.session)
      
        
        

        
  def _init_layer(self, rbm_layer_name, from_scratch = True):
    """INTENT : Initialize given layer
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    rbm_layer_name         :        name of CRBM layer that we want to initialize
    from_scratch           :        if we initialize all the variable (from_scratch is True) or not """
    
    return self.session.run(self.layer_name_to_object[rbm_layer_name].init_parameter(from_scratch))
      
        
       
        

  def _get_input_level(self, layer_level, input_data):
    
    """INTENT : Get the input from the bottom to the visible layer of the given level LAYER_LEVEL
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    layer_level         :        level of the layer we need to go from bottom up to
    input_data          :        input data for the visible layer of the bottom of the cdbn"""
    
    ret_data = input_data
    if not layer_level == 0:
      for i in range(layer_level):
        ret_layer = self.layer_name_to_object[self.layer_level_to_name[i]]
        if ret_layer.prob_maxpooling:
          ret_data = ret_layer.infer_probability(ret_data, method='forward', result = 'pooling') 
        else:
          ret_data = ret_layer.infer_probability(ret_data, method='forward', result = 'hidden') 
        if self.fully_connected_layer == i + 1:
          ret_data = tf.reshape(ret_data, [self.batch_size, -1])
          ret_data = tf.reshape(ret_data, [self.batch_size, 1, 1, ret_data.get_shape()[1].value])     
    return ret_data
      
        
        

        
  def _one_step_pretraining(self, rbm_layer_name, visible_input, n, step):
    """INTENT : Do one step of contrastive divergence for the given RBM
    ------------------------------------------------------------------------------------------------------------------------------------------
    PARAMETERS :
    rbm_layer_name         :        name of CRBM layer that we want to do one step of pretaining
    visible_input          :        configuration of the visible layer of the CRBM to train
    n                      :        length of the gibbs chain for the contrastive divergence
    step                   :        step we are at (for the learning rate decay computation)"""
    
    return self.layer_name_to_object[rbm_layer_name].do_contrastive_divergence(visible_input, n, step)
  
  
  
  
  
  def _print_error_message(self,error):
    print('----------------------------------------------')
    print('------------------ ERROR ---------------------')
    print('----------------------------------------------')
    print(error.args)
    print('----------------------------------------------')
    print('----------------------------------------------')
    print('----------------------------------------------')