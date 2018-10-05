import tensorflow as tf
import math
import numpy as np
import os

class AttWE_model(object):
    def __init__(self, lr, lambda1, lambda2, batch_size, window_size, embed_dim,
                 vocab_size, n_topic, n_neg, n_epoch, skip_step, isReg, isTopical,
                 isTest, checkpoint_dir, model_fname, sess):
        self._lr = lr
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._batch_size = batch_size
        self._mem_size = window_size*2
        self._embed_dim = embed_dim
        self._vocab_size = vocab_size
        self._n_topic = n_topic
        self._n_neg = n_neg
        self._n_epoch = n_epoch
        self._skip_step = skip_step
        self._isReg = isReg
        self._isTopical = isTopical
        self._isTest = isTest
        self.checkpoint_dir = checkpoint_dir
        self.model_fname = model_fname

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self._loss = None
        self._optimizer = None
        #self.batch_gen = batch_gen
        self.sess = sess


    def get_inputData(self):
        with tf.name_scope('input_data'):
            self.input_w = tf.placeholder(tf.int32, shape=[self._batch_size], name ='query_input')
            self.context_w = tf.placeholder(tf.int32, shape=[self._batch_size, self._mem_size], name = 'mem_input')
            self.labels = tf.placeholder(tf.int32, shape = [self._batch_size, self._vocab_size], name = 'labels')
            self.noise_w = tf.placeholder(tf.int32, shape=[self._batch_size,self._n_neg])

            if self._isTopical:
                self.input_t = tf.placeholder(tf.int32, shape=[self._batch_size], name= 'query_topic')
                self.context_t = tf.placeholder(tf.int32, shape=[self._batch_size, self._mem_size], name= 'mem_topic')
                self.labels_t = tf.placeholder(tf.int32, shape=[self._batch_size, self._n_topic])
                self.noise_t = tf.placeholder(tf.int32, shape=[self._batch_size, self._n_neg])


    # def get_analogy_data(self):
    #     self.analogy_a = tf.placeholder(dtype=tf.int32) #[n_anlogies]
    #     self.analogy_b = tf.placeholder(dtype=tf.int32)  #[n_anlogies]
    #     self.analogy_c = tf.placeholder(dtype=tf.int32)  #[n_anlogies]
    #     self.analogy_label = tf.placeholder(dtype=tf.int32) #[n_analogies]

    def inference(self):
        with tf.name_scope('mem_data'):
            weight_initializer = tf.variance_scaling_initializer()
            '''
            Aqu: embedding matrix 
            '''
            self.Aqu = tf.Variable(weight_initializer([self._vocab_size, self._embed_dim]), name='query_weights')
            self.Ain = tf.Variable(weight_initializer([self._vocab_size, self._embed_dim]), name='memIn_weights')
            self.Aout = tf.Variable(weight_initializer([self._vocab_size, self._embed_dim]), name='memOut_weights')

            self.input_em = tf.nn.embedding_lookup(self.Aqu, self.input_w) #shape = [batchsize, embedSize]
            self.mem_in_em = tf.nn.embedding_lookup(self.Ain, self.context_w) #shape=[batchsize, memSize, embedSize]
            self.mem_out_em = tf.nn.embedding_lookup(self.Aout, self.context_w) #shape=[batchsize, memSize, embedSize]
            self.noise_em = tf.nn.embedding_lookup(self.Aqu, self.noise_w)  # shape = [batchSize, n_neg, embedSize]


            if self._isTopical:
                self.Bqu = tf.Variable(weight_initializer([self._n_topic, self._embed_dim]))
                self.Bin = tf.Variable(weight_initializer([self._n_topic, self._embed_dim]))
                self.Bout = tf.Variable(weight_initializer([self._n_topic, self._embed_dim]))

                self.input_topic_em = tf.nn.embedding_lookup(self.Bqu, self.input_t) #shape = [batchsize, embedSize]
                self.mem_in_topic = tf.nn.embedding_lookup(self.Bin, self.context_t) #shape=[batchsize, memSize, embedSize]
                self.mem_out_topic = tf.nn.embedding_lookup(self.Bout, self.context_t) #shape=[batchsize, memSize, embedSize]
                self.noise_topic_em = tf.nn.embedding_lookup(self.Bqu, self.noise_t) #shape = [batchSize, n_neg, embedSize]



    # def analogy_inference(self):
    #     embed_weight = tf.nn.l2_normalize(self.Aqu, 1) #[vocab_size, embed_size]
    #
    #     analogy_a_em = tf.gather(embed_weight, self.analogy_a)
    #     analogy_b_em = tf.gather(embed_weight, self.analogy_b)
    #     analogy_c_em = tf.gather(embed_weight, self.analogy_c)
    #
    #     predicted_d_em = analogy_c_em + (analogy_b_em - analogy_a_em) #[n_anlogies, embed_size]
    #
    #     #obtain 4 words in vocab that are closest to predicted answer d
    #     #compute the cosine distance between the predicted answer and all the words in vocab
    #     similarity_dist = tf.matmul(predicted_d_em, embed_weight, transpose_b=True) #[n_analogies, vocab_size]
    #     _, self.predict_ids = tf.nn.top_k(similarity_dist,k=4)  #[n_analogies, 4]

    def build_memory(self):

        if self._isTopical:
            self.query = tf.multiply(self.input_em, self.input_topic_em)
            self.mem_inputs = tf.multiply(self.mem_in_em, self.mem_in_topic)
            self.mem_outputs = tf.multiply(self.mem_out_em, self.mem_out_topic)

            word_reshape = tf.tile(tf.expand_dims(self.input_em, 1), [1, self._n_neg, 1])  # [batchsize, n_neg, embed_dim]
            topic_reshape = tf.tile(tf.expand_dims(self.input_topic_em, 1), [1, self._n_neg, 1])  # [batchsize, n_neg, embed_dim]
            self.noise_query_w = tf.multiply(topic_reshape, self.noise_em)  # corrupting words [batchsize, n_neg, embed_dim]
            self.noise_query_t = tf.multiply(word_reshape, self.noise_topic_em)  # corrupting topics

        else:
            self.query = self.input_em
            self.mem_inputs = self.mem_in_em
            self.mem_outputs = self.mem_out_em
            self.noise_query = self.noise_em


    def generate_out_rep(self, q, isNoise=True):
        #u = tf.reshape(self.query, shape=[-1, self._embed_dim, 1])
        # n_dim = tf.size(tf.shape(q))
        # print ('hey',n_dim)
        if isNoise == False:
            u = tf.reshape(q, shape=[-1, 1, self._embed_dim])
        else:
            u = q



        p = tf.matmul(u, self.mem_inputs, transpose_b=True)  # shape=[batchSize, memSize, 1]
        #pout = tf.reshape(p, [-1, self._mem_size])  # shape=[batchSize, memSize]
        scaled_p = p / math.sqrt(self._embed_dim)  # compute the dot product of query with all memory vectors, divide each by sqrt(embed_size)
        att_score = tf.nn.softmax(scaled_p)  # shape=[batchSize, memSize]

        #att = tf.reshape(att_score, [-1, 1, self._mem_size])
        out = tf.matmul(att_score, self.mem_outputs)
        if isNoise == False:
            output_rep = tf.reshape(out, [-1, self._embed_dim])  # shape=[batchSize, embedSize]
        else:
            output_rep = out
        return output_rep

    def calc_logits(self):
        with tf.name_scope('logits'):
            outRep = self.generate_out_rep(self.query, isNoise=False)
            self.true_logits = tf.reduce_sum(tf.multiply(self.query, outRep), axis=1) #shape=[batchSize,1]

            #rep_out_noise = tf.reshape(self.rep_out, [-1, self._embed_dim, 1])

            if self._isTopical:
                #output_noisy_w = self.generate_out_rep(self.noise_query_w)
                #outRep_noisy_w = tf.reshape(output_noisy_w,[-1, self._embed_dim, 1])
                # output_noisy_t = self.generate_out_rep(self.noise_query_t)
                # outRep_noisy_t = tf.reshape(output_noisy_t, [-1, self._embed_dim, 1])
                outRep_noisy_w = self.generate_out_rep(self.noise_query_w) #[batchSize, n_neg, embedSize]
                outRep_noisy_t = self.generate_out_rep(self.noise_query_t) #[batchSize, n_neg, embedSize]

                # score_noise_w = tf.matmul(self.noise_query_w, outRep_noisy_w)
                # score_noise_t = tf.matmul(self.noise_query_t, outRep_noisy_t)
                score_noise_w = tf.reduce_sum(tf.multiply(outRep_noisy_w, self.noise_query_w), axis=1) #[batchSize*n_neg, 1]
                score_noise_t = tf.reduce_sum(tf.multiply(outRep_noisy_t, self.noise_query_t), axis=1) #[batchSize*n_neg, 1]
                self.noise_logits_w = tf.reshape(score_noise_w, [-1, self._n_neg]) #[batchSize, n_neg]
                self.noise_logits_t = tf.reshape(score_noise_t, [-1, self._n_neg])

            else:
                # output_noisy = self.generate_out_rep(self.noise_query)
                # outRep_noisy = tf.reshape(output_noisy, [-1, self._embed_dim, 1])

                outRep_noisy = self.generate_out_rep(self.noise_query) #[batchSize, n_neg, embedSize]

                #score_noise = tf.matmul(self.noise_query, outRep_noisy) #[batchSize, n_neg, 1]
                score_noise = tf.reduce_sum(tf.multiply(outRep_noisy, self.noise_query), axis=1) #[batchSize*n_neg, 1]
                self.noise_logits = tf.reshape(score_noise, [-1,self._n_neg]) #[batchSize, n_neg]


    def calc_loss(self):

        true_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.true_logits, labels=tf.ones_like(self.true_logits))

        if self._isTopical:
            noise_entropy_w = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.noise_logits_w, labels=tf.zeros_like(self.noise_logits_w))
            noise_entropy_t = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.noise_logits_t, labels=tf.zeros_like(self.noise_logits_t))
            loss_w = (tf.reduce_sum(true_entropy)+tf.reduce_sum(noise_entropy_w))/self._batch_size
            loss_t = (tf.reduce_sum(true_entropy)+tf.reduce_sum(noise_entropy_t))/self._batch_size
            loss = self._lambda1 * loss_w + self._lambda2 * loss_t
            if self._isReg:
                regularizer = tf.nn.l2_loss(self.Aqu) + tf.nn.l2_loss(self.Ain) + tf.nn.l2_loss(self.Aout) +\
                              tf.nn.l2_loss(self.Bqu) + tf.nn.l2_loss(self.Bin) + tf.nn.l2_loss(self.Bout)
                self._loss = tf.reduce_mean(loss + self._beta * regularizer)
            else:
                self._loss = loss

        else:
            noise_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.noise_logits, labels=tf.zeros_like(self.noise_logits))
            loss = (tf.reduce_sum(true_entropy)+tf.reduce_sum(noise_entropy))/self._batch_size

            if self._isReg:
                regularizer = tf.nn.l2_loss(self.Aqu) + tf.nn.l2_loss(self.Ain) + tf.nn.l2_loss(self.Aout)
                self._loss = tf.reduce_mean(loss + self._beta * regularizer)
            else:
                self._loss = loss


    def optimizer(self):
        self._optimizer = tf.train.AdamOptimizer(self._lr).minimize(self._loss)


    def build_graph(self):
        self.get_inputData()
        self.inference()
        self.build_memory()
        self.calc_logits()
        self.calc_loss()
        self.optimizer()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        # self.get_analogy_data()
        # self.analogy_inference()
        #tf.global_variables_initializer().run()






    def train(self, batch_generation, n_batch):

        epoch_loss = 0
        avg_loss = 0
        for i in range(n_batch):
            if self._isTopical:
                target_w, context_w, noise_w, target_t, context_t, noise_t = next(batch_generation)
                batch_loss, _ = self.sess.run([self._loss, self._optimizer],
                                         feed_dict={self.input_w: target_w, self.context_w: context_w,
                                                    self.noise_w: noise_w
                                             , self.input_t: target_t, self.context_t: context_t,
                                                    self.noise_t: noise_t})
                epoch_loss += batch_loss

            else:
                target_w, context_w, noise_w = next(batch_generation)
                batch_loss, _ = self.sess.run([self._loss, self._optimizer],
                                         feed_dict={self.input_w: target_w, self.context_w: context_w,
                                                    self.noise_w: noise_w})
                epoch_loss += batch_loss
                avg_loss += batch_loss
                if (i + 1) % self._skip_step == 0:
                    print('average loss at step {0} : {1}'.format(i, avg_loss / self._skip_step))
                    avg_loss = 0.0

        loss_per_epoch = epoch_loss / n_batch
        return loss_per_epoch
            # report = {'perpelexity:': math.exp(loss_per_epoch), 'epoch:': j, 'is Topical:': self._isTopical}
            # print(report)
            # epoch_loss = 0.0

    def test(self, test_batch_gen, n_batch, label='Test'):

        epoch_loss = 0
        avg_loss = 0
        for i in range(n_batch):
            if self._isTopical:
                target_w, context_w, noise_w, target_t, context_t, noise_t = next(test_batch_gen)
                batch_loss = self.sess.run([self._loss],
                                         feed_dict={self.input_w: target_w, self.context_w: context_w,
                                                    self.noise_w: noise_w
                                             , self.input_t: target_t, self.context_t: context_t,
                                                    self.noise_t: noise_t})
                epoch_loss += np.sum(batch_loss)

            else:
                target_w, context_w, noise_w = next(test_batch_gen)
                batch_loss = self.sess.run([self._loss],
                                         feed_dict={self.input_w: target_w, self.context_w: context_w,
                                                    self.noise_w: noise_w})

                epoch_loss += np.sum(batch_loss)
                avg_loss += np.sum(batch_loss)
                if (i + 1) % self._skip_step == 0:
                    print('average loss at step {0} : {1}'.format(i, avg_loss / self._skip_step))
                    avg_loss = 0.0

        loss_per_epoch = epoch_loss / n_batch
        return loss_per_epoch


    def run (self, train_batch_gen, test_batch_gen, train_n_batch, test_n_batch, valid_n_batch):
        if not self._isTest:
            for i in range(self._n_epoch):
                print ('epoch: ', i)
                train_loss = self.train(train_batch_gen, train_n_batch)
                test_loss = self.test(test_batch_gen, valid_n_batch, label='Validation')

                report_loss = {'train_loss:': train_loss,
                                'epoch: ': i,
                               'valid_loss: ': test_loss,
                               'isTopical: ': self._isTopical,
                               'isTest: ':self._isTest}
                print (report_loss)
                # if i==self._n_epoch-1:
                #
                #     report = {'perpelexity: ': math.exp(train_loss),
                #               'epoch: ': i,
                #               'valid_perplexity: ': math.exp(test_loss),
                #               'is Topical: ': self._isTopical}
                #     print(report)

                if i%10 == 0:
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_fname))


        else:
            self.load_model()

            valid_loss = self.test(train_batch_gen, valid_n_batch, label='Validation')
            test_loss = self.test(test_batch_gen, test_n_batch, label='Test')

            report_loss = {'valid_loss: ': valid_loss,
                           'test_loss: ': test_loss}


            # report = {
            #     'valid_perplexity': math.exp(valid_loss),
            #     'test_perplexity': math.exp(test_loss)
            # }
            print(report_loss)


    def load_model(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")


    # def predict_analogy(self, analogy_ques):
    #     a_analogies = analogy_ques[:,0]
    #     b_analogies = analogy_ques[:,1]
    #     c_analogies = analogy_ques[:,2]
    #     label = analogy_ques[:,3]
    #
    #     pred_id = self.sess.run([self.predict_ids], feed_dict = {self.analogy_a: a_analogies,
    #                                                              self.analogy_b: b_analogies,
    #                                                              self.analogy_c: c_analogies,
    #                                                              self.analogy_label: label})
    #
    #     return pred_id, label
    #
    #
    # def eval_analogy(self, analogy_question):
    #     correct = 0
    #     n_analogies = analogy_question.shape[0]
    #     predicted_ids, label = self.predict_analogy(analogy_question)
    #     for i in range(n_analogies):
    #         if label[i] in predicted_ids[i]:
    #             correct+=1
    #
    #     # for i in range(n_analogies):
    #     #     for j in range(4):
    #     #         if self._analogy_ques[i,3] == predicted_ids[j]:
    #     #             correct+=1
    #     #             break
    #     #         elif predicted_ids[j] in (self._analogy_ques[i,:3]):
    #     #             continue
    #     #         else:
    #     #             continue
    #
    #     accuracy = correct/n_analogies
    #     print ("answered %s out of %s analogy questions correctly with total accuracy of %s" % (correct, n_analogies,accuracy))


'''
def train(model, batch_generation):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch_loss = 0
        avg_loss = 0
        for j in range(model._n_epoch):
            for i in range(model._n_batch):
                if model._isTopical:
                    target_w, context_w, noise_w, target_t, context_t, noise_t = next(batch_generation)
                    batch_loss, _ = sess.run([model._loss, model._optimizer],
                                                  feed_dict={model.input_w: target_w, model.context_w: context_w,
                                                             model.noise_w: noise_w
                                                      , model.input_t: target_t, model.context_t: context_t,
                                                             model.noise_t: noise_t})
                    epoch_loss += batch_loss

                else:
                    target_w, context_w, noise_w = next(batch_generation)
                    batch_loss, _ = sess.run([model._loss, model._optimizer],
                                                  feed_dict={model.input_w: target_w, model.context_w: context_w,
                                                             model.noise_w: noise_w})
                    epoch_loss += batch_loss
                    avg_loss += batch_loss
                    if (i + 1) % model._skip_step == 0:
                        print('average loss at step {0} : {1}'.format(i, avg_loss / model._skip_step))
                        avg_loss = 0.0

            loss_per_epoch = epoch_loss / model._n_batch
            report = {'perpelexity:': math.exp(loss_per_epoch), 'epoch:': j, 'is Topical:': model._isTopical}
            print (report)
            epoch_loss = 0.0

'''


'''
    def train(self):
        epoch_loss = 0
        avg_loss = 0
        for i in range(self._n_batch):
            if self._isTopical:
                target_w, context_w, noise_w, target_t, context_t, noise_t = next(self.batch_gen)
                batch_loss, _ = self.sess.run([self._loss, self._optimizer],
                                              feed_dict={self.input_w: target_w, self.context_w: context_w, self.noise_w: noise_w
                                                        ,self.input_t: target_t, self.context_t: context_t, self.noise_t: noise_t})
                epoch_loss += batch_loss

            else:
                target_w, context_w, noise_w = next(self.batch_gen)
                batch_loss, _ = self.sess.run([self._loss, self._optimizer],
                                              feed_dict={self.input_w: target_w, self.context_w: context_w,
                                                         self.noise_w: noise_w})
                epoch_loss += batch_loss
                avg_loss += batch_loss
                if (i+1) % self._skip_step == 0:
                    print ('average loss at step {0} : {1}'.format(i, avg_loss/self._skip_step))
                    avg_loss = 0.0

        loss_per_epoch = epoch_loss/self._n_batch/self._batch_size
        return loss_per_epoch

    def run(self):
        for i in range(self._n_epoch):
            print('epoch {0}'.format(i))
            train_loss = self.train()
            print (train_loss)
            report = {'perpelexity:': math.exp(train_loss), 'epoch:': i, 'is Topical:': self._isTopical}
            print (report)

'''



'''
def train_AttTopical(model, n_batch, n_epochs, batch_gen, skip_step):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            print ('epoch {0}'.format(i))
            epoch_loss = 0
            for j in range(n_batch):
                target_w, context_w, noise_w, target_t, context_t, noise_t = next(batch_gen)
                batch_loss, _ = sess.run([model.loss, model.optimizer],
                                         feed_dict={model.input_w:target_w, model.context_w:context_w, model.noise_w:noise_w
                                                    ,model.input_t: target_t, model.context_t: context_t, model.noise_t: noise_t})
                epoch_loss+= batch_loss
                if (j+1) % skip_step == 0:
                    print ('average loss epoch {0} : {1}'.format(j, epoch_loss/skip_step))
                    epoch_loss = 0.0


def train_Att(model, n_batch, n_epochs, batch_gen, skip_step):
    #n_batch = int(math.ceil(len(words) / batch_size))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            print ('epoch {0}'.format(i))
            epoch_loss = 0.0
            for j in range(n_batch):
                target_w, context_w, noise_w = next(batch_gen)
                batch_loss, _ = sess.run([model.loss, model.optimizer],
                                         feed_dict={model.input_w:target_w, model.context_w:context_w, model.noise_w:noise_w})
                epoch_loss += batch_loss
                if (j+1) % skip_step == 0:
                    print ('average loss epoch {0} : {1}'.format(j, epoch_loss/skip_step))
                    epoch_loss = 0.0

'''


















