
import tensorflow as tf

class Eval(object):

    def __init__(self, checkpoint_dir, embed_weight, sess):
        self.model_dir = checkpoint_dir
        self.sess = sess
        self.embed_name = embed_weight
        self.embeds = None
        self.model_path = None
        self.load_graph()
        self.build_eval_graph()


    def load_graph(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        self.model_path = ckpt.model_checkpoint_path
        meta_path = self.model_path + '.meta'
        self.saver = tf.train.import_meta_graph(meta_path)
        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name(self.embed_name)
        self.embeds = w1
        self.model_dir = self.model_path

    def get_analogy_data(self):
        self.analogy_a = tf.placeholder(dtype=tf.int32) #[n_anlogies]
        self.analogy_b = tf.placeholder(dtype=tf.int32)  #[n_anlogies]
        self.analogy_c = tf.placeholder(dtype=tf.int32)  #[n_anlogies]
        self.analogy_label = tf.placeholder(dtype=tf.int32) #[n_analogies]

    def analogy_inference(self):
        embed_weight = tf.nn.l2_normalize(self.embeds, 1) #[vocab_size, embed_size]
        analogy_a_em = tf.gather(embed_weight, self.analogy_a)
        analogy_b_em = tf.gather(embed_weight, self.analogy_b)
        analogy_c_em = tf.gather(embed_weight, self.analogy_c)

        predicted_d_em = analogy_c_em + (analogy_b_em - analogy_a_em) #[n_anlogies, embed_size]

        #obtain 4 words in vocab that are closest to predicted answer d
        #compute the cosine distance between the predicted answer and all the words in vocab
        similarity_dist = tf.matmul(predicted_d_em, embed_weight, transpose_b=True) #[n_analogies, vocab_size]
        _, self.predict_ids = tf.nn.top_k(similarity_dist,k=4)  #[n_analogies, 4]


    def build_eval_graph(self):
        self.get_analogy_data()
        self.analogy_inference()

    def predict_analogy(self, analogy_ques):
        self.saver.restore(self.sess, self.model_path)
        self.embeds.eval()
        a_analogies = analogy_ques[:,0]
        b_analogies = analogy_ques[:,1]
        c_analogies = analogy_ques[:,2]
        label = analogy_ques[:,3]

        pred_id, = self.sess.run([self.predict_ids], feed_dict = {self.analogy_a: a_analogies,
                                                                 self.analogy_b: b_analogies,
                                                                 self.analogy_c: c_analogies,
                                                                 self.analogy_label: label})
        return pred_id, label


    def eval_analogy(self, analogy_question):
        correct = 0
        n_analogies = analogy_question.shape[0]
        start =0
        while start<n_analogies:
            end = start+2500
            if end>n_analogies: end = n_analogies
            sub_analogies = analogy_question[start:end,:]
            predicted_ids, label = self.predict_analogy(sub_analogies)
            start = end
        # for i in range(n_analogies):
        #     if label[i] in predicted_ids[i]:
        #         correct+=1

            for i in range(sub_analogies.shape[0]):
                for j in range(4):
                    if label[i] == predicted_ids[i][j]:
                        correct+=1
                        break
                    elif predicted_ids[j] in (sub_analogies[i,:3]):
                        continue
                    else:
                        continue

        accuracy = correct/n_analogies
        print ("answered %s out of %s analogy questions correctly with total accuracy of %s" % (correct, n_analogies,accuracy))






 








