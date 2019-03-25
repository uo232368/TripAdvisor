# -*- coding: utf-8 -*-
from ModelV4 import *

########################################################################################################################

class ModelV4_DEEP(ModelV4):

    def __init__(self,city,option,config,date,seed = 2):
        #Se llama igual,pero el modelo es diferente
        ModelV4.__init__(self,city,option,config,date,seed = seed)
        self.MODEL_NAME = "modelv4_deep"
        print(" "+self.MODEL_NAME)
        print("#"*50)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']

            # Número global de iteraciones
            global_step_bin = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_bin')

            # Datos de entrada -----------------------------------------------------------------------------------------------------------
            dropout = tf.placeholder_with_default(1.0, shape=(),name='dropout')
            is_train = tf.placeholder(tf.bool, name="is_train");

            user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            rest_input = tf.placeholder(tf.int32, shape=[None, 1], name="rest_input")
            rest_input_worst = tf.placeholder(tf.int32, shape=[None, 1], name="rest_input_worst")

            img_input_best = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input_best")
            img_input_worst  = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input_worst")

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E2")
            E30 = tf.Variable(tf.truncated_normal([self.V_IMG, self.V_IMG], mean=0.0, stddev=1.0 / math.sqrt(self.V_IMG)),name="E30")
            E3 = tf.Variable(tf.truncated_normal([self.V_IMG, emb_size*2], mean=0.0, stddev=1.0 / math.sqrt(emb_size*2)),name="E3")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, rest_input[:,0])

            rest_worst_emb = tf.nn.embedding_lookup(E2, rest_input_worst[:,0])

            img_input_best = tf.nn.dropout(tf.nn.relu(tf.matmul(img_input_best, E30, name='img_input_best_h')), keep_prob=dropout)
            img_input_worst = tf.nn.dropout(tf.nn.relu(tf.matmul(img_input_worst, E30, name='img_input_worst_h')), keep_prob=dropout)

            img_emb_best = tf.matmul(img_input_best, E3, name='img_emb_best')
            img_emb_worst = tf.matmul(img_input_worst, E3, name='img_emb_worst')

            user_rest_emb = tf.concat([user_emb,rest_emb],axis=1, name="user_rest_emb")
            user_rest_emb_worst = tf.concat([user_emb,rest_worst_emb],axis=1, name="user_rest_emb_worst")

            # Cálculo de LOSS y optimizador ----------------------------------------------------------------------------------------------

            dot_best = tf.reduce_sum(tf.multiply(user_rest_emb, img_emb_best), 1, keepdims=True, name="dot_best")
            dot_worst = tf.reduce_sum(tf.multiply(user_rest_emb_worst, img_emb_worst), 1, keepdims=True, name="dot_worst")

            batch_loss = tf.math.maximum(0.0,1-(dot_best-dot_worst), name="batch_loss")
            loss = tf.reduce_sum(batch_loss)

            adam = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate'])
            train_step_bin = adam.minimize(loss=loss, global_step=global_step_bin)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph
