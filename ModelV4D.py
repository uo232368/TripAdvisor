# -*- coding: utf-8 -*-
from ModelV4 import *

########################################################################################################################

class ModelV4D(ModelV4):

    def __init__(self,city,option,config,date,seed = 2):
        #Se llama igual,pero el modelo es diferente
        ModelV4.__init__(self,city,option,config,date,seed = seed)
        self.MODEL_NAME = "modelv4d"
        print(" "+self.MODEL_NAME)
        print("#"*50)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']
            usr_emb_size=emb_size*2 #1024
            rst_emb_size=emb_size   #512
            img_emb_size=emb_size   #512

            concat_size = usr_emb_size + rst_emb_size + img_emb_size

            # Variables
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

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, usr_emb_size], mean=0.0, stddev=1.0 / math.sqrt(usr_emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, rst_emb_size], mean=0.0, stddev=1.0 / math.sqrt(rst_emb_size)),name="E2")


            E30 = tf.Variable(tf.truncated_normal([self.V_IMG, emb_size*2], mean=0.0, stddev=1.0 / math.sqrt(emb_size*2)),name="E30")
            bse30 = tf.Variable(tf.zeros(emb_size*2),name="be30")
            E31 = tf.Variable(tf.truncated_normal([emb_size*2, img_emb_size], mean=0.0, stddev=1.0 / math.sqrt(img_emb_size)),name="E31")
            bse31 = tf.Variable(tf.zeros(img_emb_size),name="be31")


            T0 = tf.Variable(tf.truncated_normal([concat_size,emb_size*2], mean=0.0, stddev=1.0 / math.sqrt(emb_size*2)),name="T0")
            T1 = tf.Variable(tf.truncated_normal([emb_size*2,emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="T1")
            T2 = tf.Variable(tf.truncated_normal([emb_size,emb_size//2], mean=0.0, stddev=1.0 / math.sqrt(emb_size//2)),name="T2")
            T3 = tf.Variable(tf.truncated_normal([emb_size//2,1], mean=0.0, stddev=1.0 / math.sqrt(1)),name="T3")

            bst0 = tf.Variable(tf.zeros(emb_size*2),name="bt0")
            bst1 = tf.Variable(tf.zeros(emb_size),name="bt1")
            bst2 = tf.Variable(tf.zeros(emb_size//2),name="bt2")


            # Operaciones entrada --------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, rest_input[:,0])
            rest_worst_emb = tf.nn.embedding_lookup(E2, rest_input_worst[:,0])

            img_input_best = tf.matmul(img_input_best, E30)+bse30
            img_input_best = tf.nn.relu(img_input_best)
            img_input_best = tf.nn.dropout(img_input_best, name='img_emb_best',keep_prob=dropout)

            img_input_worst = tf.matmul(img_input_worst, E30)+bse30
            img_input_worst = tf.nn.relu(img_input_worst)
            img_input_worst = tf.nn.dropout(img_input_worst, name='img_input_worst', keep_prob=dropout)

            img_emb_best = tf.matmul(img_input_best, E31)+bse31
            img_emb_best = tf.nn.relu(img_emb_best,  name='img_emb_best')

            img_emb_worst = tf.matmul(img_input_worst, E31)+bse31
            img_emb_worst = tf.nn.relu(img_emb_worst, name='img_emb_worst')

            # Operaciones capas ocultas --------------------------------------------------------------------------------------------

            best_concat = tf.concat([user_emb,rest_emb,img_emb_best],axis=1)
            best_concat = tf.nn.dropout(best_concat, name='best_concat', keep_prob=dropout)

            worst_concat = tf.concat([user_emb,rest_worst_emb,img_emb_worst],axis=1)
            worst_concat = tf.nn.dropout(worst_concat, name='worst_concat', keep_prob=dropout)

            bt0 = tf.nn.dropout(tf.nn.relu(tf.matmul(best_concat, T0)+bst0),keep_prob=dropout, name='bt0')
            bt1 = tf.nn.dropout(tf.nn.relu(tf.matmul(bt0, T1)+bst1),keep_prob=dropout, name='bt1')
            bt2 = tf.nn.dropout(tf.nn.relu(tf.matmul(bt1, T2)+bst2),keep_prob=dropout, name='bt2')

            #bt3 = tf.nn.dropout(tf.nn.relu(tf.matmul(bt2, T3)+bst3),keep_prob=dropout, name='bt3')
            #best_out = tf.matmul(bt3, T4, name='dot_best')

            best_out = tf.matmul(bt2, T3, name='dot_best')

            #bt2 = tf.nn.dropout(tf.nn.relu(tf.matmul(bt0, T12) + bst12), keep_prob=dropout, name='bt2')

            wt0 = tf.nn.dropout(tf.nn.relu(tf.matmul(worst_concat, T0)+bst0),keep_prob=dropout,name='wt0')
            wt1 = tf.nn.dropout(tf.nn.relu(tf.matmul(wt0, T1)+bst1),keep_prob=dropout,name='wt1')
            wt2 = tf.nn.dropout(tf.nn.relu(tf.matmul(wt1, T2)+bst2),keep_prob=dropout,name='wt2')

            #wt3 = tf.nn.dropout(tf.nn.relu(tf.matmul(wt2, T3)+bst3),keep_prob=dropout,name='wt3')
            #worst_out = tf.matmul(wt3, T4, name='dot_worst')

            worst_out = tf.matmul(wt2, T3, name='dot_worst')

            #wt2 = tf.nn.dropout(tf.nn.relu(tf.matmul(wt0, T12) + bst12), keep_prob=dropout, name='wt2')

            # Cálculo de LOSS y optimizador ----------------------------------------------------------------------------------------------

            batch_loss = tf.math.maximum(0.0,1-(best_out-worst_out), name="batch_loss")
            loss = tf.reduce_sum(batch_loss)

            adam = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate'])
            train_step_bin = adam.minimize(loss=loss, global_step=global_step_bin)

            '''
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                adam = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate'])
                train_step_bin = adam.minimize(loss=loss, global_step=global_step_bin)
            '''

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph
