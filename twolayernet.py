import numpy as np
import tensorflow as tf
import math


class twolayernet():

    def __init__(self):
        self.batch_size = 1
        self.num_features = 4
        self.hidden_layer_1 = 10
        self.num_classes = 3
        self.threshold = 0.5
        self.index = 0
        self.training_loss = 0
        pass


    def graph(self):
        #Define the graph for the 2layernet binary classifier
        self.inputs = tf.placeholder(shape=(self.batch_size, self.num_features), dtype=tf.float32)
        self.output = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)

        weights_layer_1 = tf.Variable(tf.random_uniform((self.num_features, self.hidden_layer_1), -1,1), dtype=tf.float32)
        weights_output = tf.Variable(tf.random_uniform(shape=(self.hidden_layer_1, self.num_classes), minval=-1, maxval=1),
                                     dtype=tf.float32)
        bias_layer_1 = tf.Variable(tf.zeros(shape=(self.hidden_layer_1), dtype=tf.float32))
        hidden_layer_1 = tf.nn.sigmoid((tf.add(tf.matmul(self.inputs, weights_layer_1), bias_layer_1)))

        self.output_score = tf.matmul(hidden_layer_1, weights_output)     #BxC


    def calculate_loss_and_optimize(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.output, depth=self.num_classes), logits=self.output_score))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)


    def get_prediction(self, score):
        self.prediction = np.argmax(score)


    def get_batch(self, X_train, y_train):
        X_train_batch = X_train[self.index: min(len(X_train),self.index + self.batch_size)]
        y_train_batch = y_train[self.index: min(len(y_train), self.index + self.batch_size)]
        print y_train_batch
        self.index += self.batch_size
        return X_train_batch, y_train_batch


    def fit(self, X_train, y_train):
        #input is a sequence of inputs of size f each
        self.graph()
        self.calculate_loss_and_optimize()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            num_iterations = int(math.floor(float(len(X_train)) / self.batch_size))
            for itr in range(num_iterations):
                batch_x, batch_y = self.get_batch(X_train, y_train)
                feed_dict = {self.inputs:batch_x, self.output:batch_y}
                os, l,_ = session.run([self.output_score, self.loss, self.optimizer], feed_dict=feed_dict)
                self.training_loss += l
                if (itr + 1) % 10 == 0:
                    print 'Number of iteration done: {0}/{1}'.format(itr+1, num_iterations)
                    print 'Average loss for {0} iterations: {1}'.format(itr+1, self.training_loss / itr + 1)
            print 'Training Complete'
            print 'Total training loss: {}'.format(self.training_loss / num_iterations)


    def predict(self, X_test, y_test):
        self.graph()
        self.index = 0
        outputs = []
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            num_iterations = int(math.floor(float(len(X_test)) / self.batch_size))
            for itr in range(num_iterations):
                X_test_batch, y_test_batch = self.get_batch(X_test, y_test)
                score = session.run([self.output_score],feed_dict={self.inputs:X_test_batch, self.output:y_test_batch})
                self.get_prediction(score)
                outputs.extend([self.prediction])

        accuracy = np.mean(np.array(outputs) == y_test)
        return outputs, accuracy




if __name__=='__main__':

    X_train = np.array([[1,2,3,4],[1,3,4,2],[1,3,4,5],[1,2,4,5]])
    y_train = np.array([0,1,0,2])

    model = twolayernet()
    model.fit(X_train, y_train)
    output, accuracy = model.predict(X_train, y_train)
    print 'output: {0}\t accuracy:{1}'.format(output, accuracy)

