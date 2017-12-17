"""
==========================================================
Written by Dominic Masters, Graphcore, dominic.masters@graphcore.ai
==========================================================
"""
import tensorflow as tf
from timeit import default_timer as timer
import time
import random as random
import numpy as np
from six.moves import cPickle as pickle
import sys
import os
import Tools as N

def accuracy(logits,labels):
  with tf.name_scope('accuracy'):
    if labels.get_shape().ndims<2:
      labels=tf.one_hot(labels, logits.get_shape().as_list()[1])
    predictions = tf.nn.softmax(logits)
    cp=tf.equal(tf.argmax(predictions, axis=1),tf.argmax(labels, axis=1))
    accuracy=100*tf.reduce_mean(tf.cast(cp, tf.float32))
    # Top 5 prediction reports true & true for two identical predictions - this can cause incorrect results
    # - Added random noise to eliminate ties
    cp5=tf.nn.in_top_k(tf.cast(predictions, tf.float32) + tf.random_normal(predictions.get_shape(),stddev=1e-10), tf.argmax(labels, axis=1), 5)

    accuracy5=100*tf.reduce_mean(tf.cast(cp5, tf.float32))
  return accuracy, accuracy5

def create_logger(save_path):
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  class Logger(object):
    def __init__(self):
      if hasattr(sys.stdout, 'terminal'):
        self.terminal = sys.stdout.terminal
      else:
        self.terminal = sys.stdout
      self.log = open(save_path+"/log.txt", "a+")

    def write(self, message):
      self.terminal.write(message)
      self.log.write(message)

    def flush(self):
      #this flush method is needed for python 3 compatibility.
      #this handles the flush command by doing nothing.
      #you might want to specify some extra behavior here.
      pass

  sys.stdout = Logger()


class TF_session(object):
  # Tensorflow session class

  def __init__(self,dataset,inference, learning_rate,
        optimiser={'fun':tf.train.GradientDescentOptimizer, 'args':[], 'nargs':{}},
        augmentation_fun=None,
        valid_batches=10,
        train_batches=50,
        calc_train_acc=False,
        calc_valid_acc=False,
        log_dir='./logs/',
        train_sum_freq='all',
        save_graph=False,
        log_device_placement=False,
        check_divergence=True,
        name=''):

    self.dataset=dataset
    self.inference=inference
    self.learning_rate=learning_rate
    self.optimiser=optimiser
    if type(self.optimiser) is not dict:
      self.optimiser={'fun':self.optimiser}
    if not 'args' in self.optimiser.keys(): self.optimiser['args']=[]
    if not 'nargs' in self.optimiser.keys(): self.optimiser['nargs']={}

    self.augmentation_fun=augmentation_fun

    self.calc_train_acc=calc_train_acc
    self.calc_valid_acc=calc_valid_acc

    self.log_dir=log_dir
    self.train_sum_freq=train_sum_freq
    self.valid_batches=valid_batches
    self.train_batches=train_batches
    self.save_graph=save_graph

    self.check_divergence=check_divergence

    self.global_step=0

    if name=='':
      name = time.strftime("%Y%m%d-%H%M%S")
    else:
      name = time.strftime("%Y%m%d-%H%M%S") + '_' + name

    save_path=log_dir + name
    self.save_path=save_path
    self.name=name

    create_logger(self.save_path)

    if sys.version_info < (3, 0):
      print('\nTraining parameters:')
      for k, v in list(locals().iteritems()):
        if not k=='DATA' and not k=='inference':
          print(k + ' = ' + str(v))

    self._create_graph()
    self.config = tf.ConfigProto()
    self.config.log_device_placement=log_device_placement
    self.config.gpu_options.allow_growth = True

    self.train_writer = tf.summary.FileWriter(self.save_path)
    if self.save_graph:
      self.train_writer.add_graph(graph=self.graph)

    self.Results=list()

  def _train(self, total_loss, tf_global_step):
    lr = tf.constant(0.1)
    tf.summary.scalar('learning_rate', lr, collections=["TRAINING_SUMMARIES"])
    optimiser=self.optimiser['fun'](lr ,*self.optimiser['args'],**self.optimiser['nargs'])
    grads = optimiser.compute_gradients(total_loss)

    apply_gradient_op = optimiser.apply_gradients(grads)

    increment_global_step_op = tf.assign(tf_global_step, tf_global_step+1)

    with tf.control_dependencies([apply_gradient_op, increment_global_step_op]):
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # this ensures that the BN parameters are updated properly
      with tf.control_dependencies(update_ops):
        train_op = tf.no_op(name='train')

    return train_op

  def _create_graph(self):

    self.graph = tf.Graph()
    with self.graph.as_default():
      tf_global_step = tf.contrib.framework.get_or_create_global_step()
      self.dataset.initialise_tf_vars(self.calc_train_acc, self.calc_valid_acc, False)

      try:
        self.dataset.tf_train_dataset
      except AttributeError:
        print('Cannot test training set accuracy, dataset does not exist')
        self.calc_train_acc = False

      try:
        self.dataset.tf_valid_dataset
      except AttributeError:
        print('Cannot test validation set accuracy, dataset does not exist')
        self.calc_valid_acc = False

      if self.augmentation_fun != None:
        batch_dataset=self.augmentation_fun(self.dataset.tf_batch_dataset)
      else:
        batch_dataset=self.dataset.tf_batch_dataset

      with tf.name_scope('batch_image_after_augmentation'):
        if self.dataset.dim==4:
          images=tf.transpose(batch_dataset,[1,2,3,0])
          images=N.kernel_to_image(images,True)
          tf.summary.image('batch_images', images,collections=['IMAGE_SUMMARIES'])

      batch_logits = self.inference(batch_dataset, self.dataset.num_classes, training=True,reuse=False)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=batch_logits, labels=self.dataset.tf_batch_labels)
      self.batch_xent = tf.reduce_mean(cross_entropy)
      tf.add_to_collection('losses', self.batch_xent)
      losses=tf.get_collection('losses')
      self.loss=tf.add_n(losses, name='total_loss')

      self.batch_accuracy, self.batch_accuracy5 = accuracy(batch_logits,self.dataset.tf_batch_labels)
      tf.summary.scalar('loss', self.loss, collections=["TRAINING_SUMMARIES"])
      tf.summary.scalar('cross_entropy/batch', self.batch_xent, collections=["TRAINING_SUMMARIES"])
      tf.summary.scalar('accuracy/batch', self.batch_accuracy, collections=["TRAINING_SUMMARIES"])
      tf.summary.scalar('accuracy/batch5', self.batch_accuracy5, collections=["TRAINING_SUMMARIES"])
      N.variable_summaries(cross_entropy,'cross_entropy_batch')

      if self.calc_train_acc:
        train_logits=list()
        sp_train_dataset= tf.split(self.dataset.tf_train_dataset, self.train_batches, axis = 0)
        for i in range(len(sp_train_dataset)):
          with tf.control_dependencies(train_logits):
            train_logits.append(self.inference(sp_train_dataset[i], self.dataset.num_classes, training=False,reuse=True))
        cat_train_logits=tf.concat(train_logits,axis=0)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=cat_train_logits, labels=self.dataset.tf_train_labels)
        self.train_xent = tf.reduce_mean(cross_entropy)
        self.train_accuracy, self.train_accuracy5= accuracy(cat_train_logits,self.dataset.tf_train_labels)
        tf.summary.scalar('cross_entropy/train', self.train_xent, collections=["TRAINING_SUMMARIES"])
        tf.summary.scalar('accuracy/train', self.train_accuracy, collections=["TRAINING_SUMMARIES"])
        tf.summary.scalar('accuracy/train5', self.train_accuracy5, collections=["TRAINING_SUMMARIES"])
        N.variable_summaries(cross_entropy,'cross_entropy_train')
      else:
        self.train_xent=tf.constant(0)
        self.train_accuracy=tf.constant(0)
        self.train_accuracy5=tf.constant(0)

      if self.calc_valid_acc:
        valid_logits=list()
        sp_valid_dataset= tf.split(self.dataset.tf_valid_dataset, self.valid_batches, axis = 0)
        for i in range(len(sp_valid_dataset)):
          with tf.control_dependencies(valid_logits):
            valid_logits.append(self.inference(sp_valid_dataset[i], self.dataset.num_classes, training=False,reuse=True))
        cat_valid_logits=tf.concat(valid_logits,axis=0)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=cat_valid_logits, labels=self.dataset.tf_valid_labels)
        self.valid_xent = tf.reduce_mean(cross_entropy)
        self.valid_accuracy, self.valid_accuracy5= accuracy(cat_valid_logits,self.dataset.tf_valid_labels)
        tf.summary.scalar('cross_entropy/valid', self.valid_xent, collections=["TRAINING_SUMMARIES"])
        tf.summary.scalar('accuracy/valid', self.valid_accuracy, collections=["TRAINING_SUMMARIES"])
        tf.summary.scalar('accuracy/valid5', self.valid_accuracy5, collections=["TRAINING_SUMMARIES"])
        N.variable_summaries(cross_entropy,'cross_entropy_valid')
      else:
        self.valid_xent=tf.constant(0)
        self.valid_accuracy=tf.constant(0)
        self.valid_accuracy5=tf.constant(0)

      print('')

      self.train_op = self._train(self.loss, tf_global_step)

      self.training_summary_op = tf.summary.merge(tf.get_collection('TRAINING_SUMMARIES'))

  def run(self,iterations,log_freq=500):
    with tf.Session(graph=self.graph,config=self.config) as sess:
      self._print_trainable_variables()

      tf.global_variables_initializer().run(session=sess)
      run_options=None
      run_metadata=None

      start = timer()
      header=str()
      header1=str()
      format_str=str()
      header+="%9s %8s %8s"%('Iteration','Epoch','Loss')
      header1+="%9s %8s %8s"%(' ',' ',' ')
      header+=" | %9s %8s %9s"%('Cross_ent','Accuracy','Accuracy5')
      header1+=" | %9s %8s %9s"%(' ',' Batch  ',' ')
      format_str+="%(Iterations)9d %(Epoch)8.1f %(Loss)8.5f"
      format_str+=" | %(Batch_Cross_Entropy)9.5f %(Batch_Accuracy)8.3f %(Batch_Accuracy5)9.3f"
      if self.calc_train_acc:
        header+=" | %9s %8s %9s"%('Cross_ent','Accuracy','Accuracy5')
        header1+=" | %9s %8s %9s"%(' ','Training',' ')
        format_str+=" | %(Train_Cross_Entropy)9.5f %(Train_Accuracy)8.3f %(Train_Accuracy5)9.3f"
      if self.calc_valid_acc:
        header+=" | %9s %8s %9s"%('Cross_ent','Accuracy','Accuracy5')
        header1+=" | %8s %10s %8s"%(' ','Validation',' ')
        format_str+=" | %(Valid_Cross_Entropy)9.5f %(Valid_Accuracy)8.3f %(Valid_Accuracy5)9.3f"
      header+=" | %10s %8s %10s %8s"%('Batch','Ex/sec','Major_It','Total')
      header1+=" | %13s %11s %13s"%('','Time (secs)', '')
      format_str+=" | %(Batch_Time)10.7f %(Ex_per_sec)8d %(Maj_Time)10.6f %(Total_Time)8.6g"

      print(header1)
      print(header)
      batch_times=[0]
      ex_per_batch=[0]

      for step in range(iterations):
        feed_dict = self.dataset.get_next_batch()
        print("Step: ", step)
        
        if (step % log_freq == 0) or step==iterations-1:
          summaries=list()
          if self.train_sum_freq=='all' or (self.train_sum_freq=='end' and step==iterations-1):
            summaries.append(self.training_summary_op)

          start_acc= timer()
          if step==0: # calculate loss etc without train_op
            l, batch_xent, batch_acc, batch_acc5,\
            train_xent, train_acc, train_acc5,\
            valid_xent, valid_acc, valid_acc5, summary=sess.run([
              self.loss, self.batch_xent, self.batch_accuracy, self.batch_accuracy5,
              self.train_xent, self.train_accuracy, self.train_accuracy5,
              self.valid_xent, self.valid_accuracy, self.valid_accuracy5, summaries],
              feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
          else:
            _, l, batch_xent, batch_acc, batch_acc5,\
            train_xent, train_acc, train_acc5,\
            valid_xent, valid_acc, valid_acc5, summary=sess.run([self.train_op,
              self.loss, self.batch_xent, self.batch_accuracy, self.batch_accuracy5,
              self.train_xent, self.train_accuracy, self.train_accuracy5,
              self.valid_xent, self.valid_accuracy, self.valid_accuracy5, summaries],
              feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

          end_acc= timer()
          end = timer()

          epoch=float(self.global_step*self.dataset.batch_size)/float(self.dataset.num_train_ex)

          for s in summary:
            if s != None:
              self.train_writer.add_summary(s, self.global_step)
          acc_time=(end_acc - start_acc)
          tot_time=(end - start)

          RES={'Iterations':step, 'Epoch': epoch, 'Loss': l,
            'Batch_Cross_Entropy': batch_xent, 'Batch_Accuracy': batch_acc, 'Batch_Accuracy5': batch_acc5}
          if self.calc_train_acc:
            RES.update({'Train_Cross_Entropy': train_xent, 'Train_Accuracy': train_acc, 'Train_Accuracy5': train_acc5})
          if self.calc_valid_acc:
            RES.update({'Valid_Cross_Entropy': valid_xent, 'Valid_Accuracy': valid_acc, 'Valid_Accuracy5': valid_acc5})
          RES.update({'Batch_Time': np.mean(batch_times), 'Ex_per_sec': np.mean(ex_per_batch),
            'Maj_Time': acc_time, 'Total_Time': tot_time})

          print(format_str%RES)

          self.Results.append(RES)

          if self.check_divergence:
            if np.isnan(l):
              print("Loss is NaN")
              break
            if step > (0.25 * iterations):
              if train_acc < (100/self.dataset.num_classes)*1.5:
                print("Training accuracy is near random - stopped due to failure to train")
                break
              if valid_acc < (100/self.dataset.num_classes)*1.5:
                print("Validation accuracy is near random - stopped due to failure to train")
                break

        else:
          start_it = timer()
          _=sess.run([self.train_op],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
          end_it = timer()
          batch_times.extend([end_it - start_it])
          ex_per_batch.extend([self.dataset.batch_size/batch_times[-1]])

          if len(batch_times)>100: #running average of batch calculation times
            batch_times=batch_times[-100:]
            ex_per_batch=ex_per_batch[-100:]

        self.global_step += 1

    pickle.dump( self.Results, open( self.save_path + "/Results.pickle", "wb" ) )
    self.train_writer.close()

  def _print_trainable_variables(self):
    # print paramater summary
    print('\nTrainable Variables:')
    total_parameters = 0
    for variable in tf.trainable_variables():
      print(variable)
      variable_parameters = 1
      for DIM in variable.get_shape():
        variable_parameters *= DIM.value
      total_parameters += variable_parameters
    print('Total Parameters:' + str(total_parameters) + '\n')

  def plot_results(self):
    import matplotlib.pyplot as plt #imported here to stop crash on some server machines
    plt.figure()
    plt.plot(self.Results['steps'],self.Results['train_acc'], label='Training Accuracy (Top 1)')
    plt.plot(self.Results['steps'],self.Results['valid_acc'], label='Validation Accuracy (Top 1)')
    plt.plot(self.Results['steps'],self.Results['train_acc5'], label='Training Accuracy (Top 5)')
    plt.plot(self.Results['steps'],self.Results['valid_acc5'], label='Validation Accuracy (Top 5)')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.show()
















