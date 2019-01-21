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
            dpout = tf.placeholder(tf.float32, name='dpout')

            # Datos de entrada -----------------------------------------------------------------------------------------------------------
            # Array del tamaño del batch con las X
            user_rest_input = tf.placeholder(tf.int32, shape=[None, 2], name="user_rest_input")
            img_input = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input")

            # Capas de salida -----------------------------------------------------------------------------------------------------------

            bin_labels = tf.placeholder(tf.float32, shape=[None, 1], name="bin_labels")

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E2")

            # Salida ---------------------------------------------------------------------------------------------------------------------

            R0 = tf.Variable(tf.truncated_normal([concat_size, hidden_size], mean=0.0, stddev=1.0 / math.sqrt(hidden_size)), name="R0")
            b0 = tf.Variable(tf.zeros([hidden_size]), name="b0")

            R1 = tf.Variable(tf.truncated_normal([hidden_size, hidden2_size], mean=0.0, stddev=1.0 / math.sqrt(hidden2_size)),name="R1")
            b1 = tf.Variable(tf.zeros([hidden2_size]), name="b1")

            R2 = tf.Variable(tf.truncated_normal([hidden2_size, 1], mean=0.0, stddev=1.0 / math.sqrt(1)),name="R2")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_rest_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, user_rest_input[:,1])

            user_emb = tf.nn.dropout(user_emb,keep_prob=dpout);
            rest_emb = tf.nn.dropout(rest_emb,keep_prob=dpout);
            img_input = tf.nn.dropout(img_input,keep_prob=dpout);

            c1 = tf.concat([user_emb,rest_emb, img_input],axis=1, name="concat_r1")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            h1 = tf.matmul(c1, R0, name='h1')+b0
            h1 = tf.nn.dropout(h1, keep_prob=dpout);
            h1 = tf.nn.relu(h1);

            h2 = tf.matmul(h1, R1, name='h2')+b1
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

        tss = time.time()

        train_bin_loss = []
        train_bin_batches = np.array_split(self.TRAIN_V3, len(self.TRAIN_V3) // self.CONFIG['batch_size'])

        for bn in range(len(train_bin_batches)):

            batch_train_bin = train_bin_batches[bn][['id_user', 'id_restaurant','vector','like']].values

            # Si se entrena con las imágenes
            if (self.CONFIG['use_images'] == 1):

                feed_dict_bin = {"user_rest_input:0": batch_train_bin[:, [0, 1]],
                                 "img_input:0": np.row_stack(batch_train_bin[:, [2]][:, 0]),
                                 "bin_labels:0": batch_train_bin[:, [3]],
                                 'dpout:0': self.CONFIG['dropout']}
            else:

                feed_dict_bin = {"user_rest_input:0": batch_train_bin[:, [0, 1]],
                                 "img_input:0": np.zeros((len(batch_train_bin),self.V_IMG)),
                                 "bin_labels:0": batch_train_bin[:, [3]],
                                 'dpout:0': self.CONFIG['dropout']}

            _, batch_softplus,batch_bin_prob = self.SESSION.run(['train_step_bin:0', 'batch_softplus:0','batch_bin_prob:0'], feed_dict=feed_dict_bin)


            train_bin_loss.extend(batch_softplus[:, 0])


        ''' 
        #LA F1 DEL ULTIMO BATCH
        
        f = np.vectorize(lambda x: 0 if(x<.5) else 1)
        real = train_bin_batches[bn].like.values
        pred = f(np.array(batch_bin_prob[:,0]))
        print(metrics.f1_score(real,pred))
        '''

        #print("Train time: "+str(time.time()-tss))

        return [np.mean(train_bin_loss)]

    def dev(self):

        tss = time.time()

        dev_bin_res = pd.DataFrame()
        dev_loss = []

        for batch_d in np.array_split(self.DEV, 30):
            batch_dtfm = batch_d.copy()
            batch_dtfm = batch_dtfm[['id_user', 'id_restaurant', 'like']]

            batch_dm = batch_dtfm.values
            feed_dict = {"user_rest_input:0": batch_dm[:, [0, 1]],
                         "img_input:0": np.zeros((len(batch_d), self.V_IMG)),
                         "bin_labels:0": batch_dm[:, [2]],
                         'dpout:0': 1.0}

            batch_bin_prob, batch_softplus = self.SESSION.run(['batch_bin_prob:0', 'batch_softplus:0'], feed_dict=feed_dict)

            batch_dtfm['prediction'] = batch_bin_prob[:, 0]
            dev_loss.extend(batch_softplus[:, 0])

            dev_bin_res = dev_bin_res.append(batch_dtfm, ignore_index=True)

        #print("DEV time 1: "+str(time.time()-tss))

        hits, avg_pos, median_pos = self.getTopN(dev_bin_res, data=self.DEV)
        hits = list(hits.values())

        return ((np.average(dev_loss), hits, avg_pos, median_pos), avg_pos)

    def gridSearchPrint(self,epoch,train,dev):

        log_items = [epoch+1]

        log_items.extend(np.round([train[0], dev[0]], decimals=4))
        log_items.append(bcolors.OKBLUE)
        log_items.extend(np.round(dev[1], decimals=4))
        log_items.append(bcolors.ENDC)

        log_items.extend(np.round([dev[2], dev[3]], decimals=4))
        log_line = "\t".join(map(lambda x: str(x), log_items))
        log_line = log_line.replace("\t\t", "\t")

        print(log_line)

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()
