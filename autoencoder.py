import tensorflow as tf
import numpy as np
import utils
import math
from datetime import datetime
import sys

'''
The model looks like this --
encoder = RNN
    -- cell = GRU
    -- type = bidirectional
    -- hidden units = 150
    -- inputs = batch padded
    -- input format = time major
    -- vocab size = V
    -- embedding size = 300

decoder = RNN
    -- cell = GRU
    -- type = forward
    -- hidden units = 300
    -- inputs = batch padded = input of encoder
    -- input format = time major
    -- vocab size = V

structure =
    Training: input batch --> encoder --> context vector --> decoder --> output --> loss = output - input --> optimizer
    Testing: input batch --> encoder --> context vector
'''


class autoencoder:

    def __init__(self, vocab, embedding_size, en_hidden_state, dec_hidden_state, batch_size, learning_rate, no_epochs=0, validate=False, dropout=1.0):
        #The entire model will be initialized in the init part
        self.vocab_size = vocab
        self.embedding_size = embedding_size
        self.enc_hidden_state = en_hidden_state
        self.dec_hidden_state = dec_hidden_state
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.PAD = 0
        self.EOS = 1
        self.max_time_step = 0
        self.epoch_loss = []
        self.test_loss = 0
        self.no_epochs = no_epochs
        self.iteration = 0
        self.val_loss = []
        self.validate = validate


        self.encoder_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.decoder_input = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.decoder_target = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.encoder_lengths = tf.placeholder(shape=[None, ], dtype=tf.int32)

        initializer = tf.contrib.layers.xavier_initializer()
        self.decoder_weights = tf.get_variable("self.decoder_weights", shape=[self.dec_hidden_state, self.vocab_size], initializer=initializer)
        self.decoder_bias = tf.Variable(tf.zeros(shape=[self.vocab_size]), dtype=tf.float32)
        self.embeddings = tf.get_variable("self.embeddings", shape=[self.vocab_size, self.embedding_size], initializer=initializer)
        self.keep_prob = tf.placeholder(tf.float32)

        assert self.dec_hidden_state == 2*self.enc_hidden_state

    def create_graph(self):
        #Encoder:
        enc_inputs = tf.nn.embedding_lookup(self.embeddings, self.encoder_input)
        with tf.variable_scope('encoder'):
            encoder_cell_fw = tf.contrib.rnn.GRUCell(num_units=self.enc_hidden_state)
            encoder_cell_fw = tf.contrib.rnn.DropoutWrapper(encoder_cell_fw, output_keep_prob=self.keep_prob)
            encoder_cell_bw = tf.contrib.rnn.GRUCell(num_units=self.enc_hidden_state)
            encoder_cell_bw = tf.contrib.rnn.DropoutWrapper(encoder_cell_bw, output_keep_prob=self.keep_prob)
        inital_state = encoder_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        ((enc_output_fw, enc_output_bw), (enc_final_state_fw, enc_final_state_bw)) = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw, encoder_cell_bw, enc_inputs, self.encoder_lengths, initial_state_fw=inital_state, initial_state_bw=inital_state, time_major=True)
        self.enc_output_states = tf.concat((enc_output_fw, enc_output_bw),2)         #Final outputs are of the shape TxBx2H
        self.enc_final_state = tf.concat((enc_final_state_fw, enc_final_state_bw),1) #Final state is of the shape Bx2H

        #Decoder:
        dec_inputs = tf.nn.embedding_lookup(self.embeddings, self.decoder_input)
        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.GRUCell(num_units=self.dec_hidden_state)
        self.decoder_output, self.dec_final_state = tf.nn.dynamic_rnn(decoder_cell, dec_inputs, initial_state=self.enc_final_state, time_major=True, dtype=tf.float32)


    def calculate_loss_and_optimize(self):
        #Loss and Optimizer
        dec_time, dec_batch, dec_dim = tf.unstack(tf.shape(self.decoder_output))
        flattened_decoder_output = tf.reshape(self.decoder_output, (-1, dec_dim))
        final_output = tf.add(tf.matmul(flattened_decoder_output, self.decoder_weights), self.decoder_bias)
        self.decoder_logits = tf.reshape(final_output, (dec_time, dec_batch, self.vocab_size))
        self.decoder_predictions = tf.argmax(self.decoder_logits, 2)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.decoder_target, depth=self.vocab_size, dtype=tf.float32), logits=self.decoder_logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def prepare_graph(self):

        self.create_graph()
        self.calculate_loss_and_optimize()


    def get_batch(self, enc_input):

        enc_input_batch, enc_input_lengths, max_step = utils.get_batch_padded(enc_input, self.batch_size)
        assert len(enc_input_batch) == self.batch_size
        dec_target_batch = enc_input_batch
        dec_input_batch = np.ones(shape=enc_input_batch.shape, dtype=np.int32)      #This ensures that the first input is an EOS
        for i, seq in enumerate(dec_target_batch):
            dec_input_batch[i][1:] = seq[:-1]
            #if i == 0:
            #    print 'sample encoder input: \n{}'.format(enc_input_batch[0])
            #    print 'sample decoder input: \n{}'.format(dec_input_batch[0])
            #    print 'sample decoder output: \n{}'.format(dec_target_batch[0])
        self.max_time_step = max_step
        return enc_input_batch, enc_input_lengths, dec_input_batch, dec_target_batch

    def next_iter(self, enc_input, enc_lengths, dec_input, dec_target, session, train=False):

        feed = {}
        feed[self.encoder_input] = utils.transform_time_major(enc_input, self.batch_size, self.max_time_step)
        feed[self.encoder_lengths] = enc_lengths
        feed[self.decoder_input] = utils.transform_time_major(dec_input, self.batch_size, self.max_time_step)
        feed[self.decoder_target] = utils.transform_time_major(dec_target, self.batch_size, self.max_time_step)

        if train:
            feed[self.keep_prob] = self.dropout
            context_vector, l, _ = session.run([self.enc_final_state, self.loss, self.optimizer], feed_dict=feed)
        else:
            feed[self.keep_prob] = 1.0
            context_vector, decoder_output, l = session.run([self.enc_final_state, self.decoder_predictions, self.loss], feed_dict=feed)

        return context_vector, l


    def fit(self, text_sequence, validation_sequence = None, checkpoint = 1):

        self.prepare_graph()
        print 'graph has been prepared'
        no_iterations = int(math.ceil(float(len(text_sequence)) / self.batch_size))
        print 'Number of iterations in each epoch: ', no_iterations
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            if self.validate:
                run = True
                iter_loss = 0
                t1 = datetime.now()
                while run:
                    encoder_input, encoder_lengths, decoder_input, decoder_target = self.get_batch(
                        enc_input=text_sequence)
                    _, loss = self.next_iter(encoder_input, encoder_lengths, decoder_input, decoder_target, session,
                                             train=True)
                    iter_loss += loss
                    self.iteration += 1
                    if  self.iteration % 1000 == 0:
                        t2 = datetime.now()
                        print '{0} out of {1} iterations run for epoch {2}'.format(self.iteration, no_iterations, self.no_epochs)
                        print 'time taken to run 1000 iterations: {0}'.format(t2-t1)
                        t1 = datetime.now()
               
                    if self.iteration == no_iterations:
                        self.iteration = 0
                        self.no_epochs += 1
                        print '\nTraining Loss for epoch {0}:{1}'.format(self.no_epochs, iter_loss / float(no_iterations))
                        self.epoch_loss.append(iter_loss / float(no_iterations))
                        iter_loss = 0
                        if self.no_epochs % 1 == 0:
                            val_batch_size = self.batch_size
                            no_val_iterations = int(math.ceil(float(len(validation_sequence)) / val_batch_size))
                            validation_loss = 0
                            for i in range(no_val_iterations):
                                enc_input, enc_length, dec_input, dec_target = self.get_batch(validation_sequence)
                                _, loss = self.next_iter(enc_input, enc_length, dec_input, dec_target, session,
                                                         train=False)
                                validation_loss += loss
                            self.val_loss.append(validation_loss / float(no_val_iterations))

                            print 'Validation loss for epoch {0}:{1}\n'.format(self.no_epochs, validation_loss / float(no_val_iterations))

                            saver.save(session, '/iesl/canvas/tsahay/Autoencoder/Ckpt/ae_titles_val_' + str(self.no_epochs) + '.ckpt')

                            if len(self.val_loss) >= 2 and self.val_loss[-1] > self.val_loss[-2]:
                                run = False
                                print 'Model Training Complete at epochs:{0}'.format(self.no_epochs)                         
                                best_model = '/iesl/canvas/tsahay/Autoencoder/Ckpt/ae_titles_val_' + str(self.no_epochs - checkpoint) + '.ckpt'

            else:
                st = datetime.now()
                print '\nStarting the training at: {0}\n'.format(st)
                for epoch in range(self.no_epochs):
                    iter_loss = 0
                    t1 = datetime.now()
                    for it in range(no_iterations):
                        encoder_input, encoder_lengths, decoder_input, decoder_target = self.get_batch(text_sequence)
                        _, l = self.next_iter(encoder_input, encoder_lengths, decoder_input, decoder_target, session, train=True)
                        iter_loss += l
                        if (it + 1) % 1000 == 0:
                            t2 = datetime.now()
                            print '\nAverage Iteration loss for {0}/{2} batches for epoch {3}: {1}'.format((it + 1), iter_loss / float(it + 1), no_iterations, epoch)
                            print 'Time taken for 1000 batches: {0}\n'.format(t2-t1)
                            t1 = datetime.now()
                    en = datetime.now()
                    self.epoch_loss.append(iter_loss / float(no_iterations))
                    print '\naverage epoch loss for epoch {0}/{2}:{1}'.format(epoch, iter_loss / float(no_iterations), self.no_epochs)
                    print 'Time taken: ', en-st
                    st = datetime.now() 
                    if (epoch + 1) % checkpoint == 0:
                        print 'checkpoint at epoch {0}. Saved at path: {1}\n '.format(epoch, '/iesl/canvas/tsahay/Dataset/Ckpt/ae_' + str(epoch + 1) + '.ckpt')
                        saver.save(session, '/iesl/canvas/tsahay/Dataset/Ckpt/ae_' + str(epoch + 1) + '.ckpt')

                print 'Final checkpoint saved at: {0}'.format('/iesl/canvas/tsahay/Dataset/Ckpt/ae_' + str(self.no_epochs) + '.ckpt')
                saver.save(session, '/iesl/canvas/tsahay/Dataset/Ckpt/ae_' + str(self.no_epochs) + '.ckpt')
                best_model = '/iesl/canvas/tsahay/Dataset/Ckpt/ae_' + str(self.no_epochs) + '.ckpt'

        return best_model


    def predict(self, text_sequence, model_path):
        print 'predicting results based on test data.'
        context_vectors = []
        saver = tf.train.Saver()
        no_iterations = int(math.ceil(float(len(text_sequence)) / self.batch_size))
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session, model_path)
            for iter in range(no_iterations):
                encoder_input, encoder_lengths, decoder_input, decoder_target = self.get_batch(enc_input = text_sequence)
                context_vector, loss = \
                    self.next_iter(enc_input=encoder_input, enc_lengths=encoder_lengths, dec_input=decoder_input, dec_target=decoder_target, session=session)
                self.test_loss += loss
                print 'shape of the context vector: ', np.array(context_vector).shape
                print 'shape should be - {0} x {1}'.format(self.batch_size, self.dec_hidden_state)
                context_vectors.extend(context_vector)

        return context_vectors















