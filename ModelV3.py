# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV3(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv3  (img-rec)"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']
            hidden_size = self.CONFIG['hidden_size']
            hidden2_size = self.CONFIG['hidden2_size']

            concat_size = emb_size*2 + self.V_IMG

            # Número global de iteraciones
            global_step_bin = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_bin')
            dpout = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='dpout')

            # Datos de entrada -----------------------------------------------------------------------------------------------------------
            # Array del tamaño del batch con las X
            user_rest_input = tf.placeholder(tf.float32, shape=[None, 2 + self.V_IMG], name="user_rest_img_input")

            # Capas de salida -----------------------------------------------------------------------------------------------------------

            bin_labels = tf.placeholder(tf.float32, shape=[None, 1], name="bin_labels")

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E2")

            # Salida ---------------------------------------------------------------------------------------------------------------------

            R0 = tf.Variable(tf.truncated_normal([concat_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)), name="R0")
            R1 = tf.Variable(tf.truncated_normal([hidden_size, hidden2_size], mean=0.0, stddev=1.0 / math.sqrt(hidden2_size)),name="R1")
            R2 = tf.Variable(tf.truncated_normal([hidden2_size, 1], mean=0.0, stddev=1.0 / math.sqrt(1)),name="R2")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, tf.cast(user_rest_input[:,0], tf.int32))
            rest_emb = tf.nn.embedding_lookup(E2, tf.cast(user_rest_input[:,1], tf.int32))

            user_emb = tf.nn.dropout(user_emb,keep_prob=dpout);
            rest_emb = tf.nn.dropout(rest_emb,keep_prob=dpout);

            c1 = tf.concat([user_emb,rest_emb, user_rest_input[:,2:]],axis=1, name="concat_r1")


            # Operaciones -----------------------------------------------------------------------------------------------------------------

            h1 = tf.matmul(c1, R0, name='h1')
            h1 = tf.nn.dropout(h1, keep_prob=dpout);
            h1 = tf.nn.relu(h1);

            h2 = tf.matmul(h1, R1, name='h2')
            h2 = tf.nn.dropout(h2, keep_prob=dpout);
            h2 = tf.nn.relu(h2);

            out_bin = tf.matmul(h2, R2, name='out_bin')

            # Cálculo de LOSS y optimizador ----------------------------------------------------------------------------------------------

            batch_bin_prob = tf.nn.sigmoid(out_bin, name='batch_bin_prob')
            batch_softplus = tf.nn.softplus((1 - 2 * bin_labels) * out_bin, name='batch_softplus')
            loss_softplus = tf.reduce_mean(batch_softplus, name='loss_softplus')

            # Minimizar la loss
            train_step_bin = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate']).minimize(loss=loss_softplus, global_step=global_step_bin)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph

    def train(self):
        return False

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()
