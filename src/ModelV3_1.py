# -*- coding: utf-8 -*-
from src.ModelClass import *

########################################################################################################################

class ModelV3_1(ModelClass):

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

            user_emb = tf.nn.embedding_lookup(E1, user_rest_input[:,0], name="user_emb") #user_emb/Identity:0
            rest_emb = tf.nn.embedding_lookup(E2, user_rest_input[:,1], name="rest_emb") #rest_emb/Identity:0

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

            #batch_softplus = tf.nn.sigmoid_cross_entropy_with_logits(labels=bin_labels, logits=out_bin, name='batch_softplus')
            #loss_softplus = tf.reduce_mean(batch_softplus)

            batch_softplus = tf.nn.softplus((1 - 2 * bin_labels) * out_bin, name='batch_softplus')
            loss_softplus = tf.reduce_mean(batch_softplus, name='loss_softplus')

            # Minimizar la loss
            train_step_bin = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate']).minimize(loss=loss_softplus, global_step=global_step_bin)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph

    def train(self):

        tss = time.time()

        train_img_res = pd.DataFrame()

        train_bin_loss = []
        train_bin_batches = np.array_split(self.TRAIN_V3_1, len(self.TRAIN_V3_1) // self.CONFIG['batch_size'])

        for bn in range(len(train_bin_batches)):

            batch_dtfm_imgs_ret = train_bin_batches[bn][['id_user', 'id_restaurant','vector','like']].copy()
            batch_train_bin = batch_dtfm_imgs_ret.values

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

            _, R0,E1,E2,batch_softplus,batch_bin_prob = self.SESSION.run(['train_step_bin:0','R0:0','E1:0','E2:0','batch_softplus:0','batch_bin_prob:0'], feed_dict=feed_dict_bin)
            # ue, re, im = self.SESSION.run(['user_emb/Identity:0','rest_emb/Identity:0','img_input'], feed_dict=feed_dict_bin)

            if(self.CURRENT_EPOCH==30):
                sufx = ""
                #self.matrixToImage(E1, path="out/31_01_2019/E1"+sufx)
                #self.matrixToImage(E2, path="out/31_01_2019/E2"+sufx)
                self.matrixToImage(R0, path="out/31_01_2019/R0"+sufx)

            train_bin_loss.extend(batch_softplus[:, 0])

            batch_dtfm_imgs_ret['prediction'] = batch_bin_prob[:, 0]
            train_img_res = train_img_res.append(batch_dtfm_imgs_ret, ignore_index=True)

        real = train_img_res["like"].values
        pred = (train_img_res["prediction"].values > .5) * 1

        ac = metrics.accuracy_score(real, pred)

        return (np.mean(train_bin_loss),ac)

    def dev(self):

        dev_loss = []
        dev_img_res = pd.DataFrame()

        for batch_di in np.array_split(self.DEV_V3_1, 10):
            batch_dtfm_imgs_ret = batch_di.copy()
            batch_dtfm_imgs = batch_di[['id_user', 'id_restaurant', 'vector', 'like']]
            batch_dm = batch_dtfm_imgs.values

            feed_dict = {"user_rest_input:0": batch_dm[:, [0, 1]],
                         "img_input:0": np.row_stack(batch_dm[:, [2]][:, 0]),
                         "bin_labels:0": batch_dm[:, [3]],
                         'dpout:0':  1.0}

            batch_bin_prob, batch_softplus = self.SESSION.run(['batch_bin_prob:0', 'batch_softplus:0'], feed_dict=feed_dict)
            batch_dtfm_imgs_ret['prediction'] = batch_bin_prob[:, 0]

            dev_loss.extend(batch_softplus[:, 0])

            dev_img_res = dev_img_res.append(batch_dtfm_imgs_ret, ignore_index=True)

        real = dev_img_res["like"].values
        pred = (dev_img_res["prediction"].values>.5)*1

        '''
        f1 = metrics.f1_score(real, pred)
        pc = metrics.recall_score(real, pred)
        rc = metrics.precision_score(real, pred)
        '''

        ac = metrics.accuracy_score(real, pred)

        return ((np.average(dev_loss),ac), 1-ac)

    def test(self):

        dev_img_loss = []
        dev_img_loss_rndm = []
        dev_img_loss_max = []
        dev_img_loss_min = []
        dev_img_loss_cnt = []

        def getDist(dev,r,mode = "best"):

            if (len(dev) > 1): dev = np.row_stack(dev)
            else: dev = [dev[0]]

            if(mode == "best"):
                item = r.iloc[-1,:]['vector']

            if (mode == "random"):
                item = r.sample(1)['vector'].values[0]

            if (mode == "centroid"):
                all = r.vector.values
                if (len(all) > 1): all = np.row_stack(all)
                else: all = [all[0]]

                cnt = np.mean(all, axis=0)
                dsts = scipy.spatial.distance.cdist([cnt], all, 'euclidean')
                indx = np.argmin(dsts)
                item = all[np.argmin(dsts), :]

            if (mode == "max"):
                all = r.vector.values
                if(len(all)>1):all = np.row_stack(all)
                else:all = [all[0]]

                dsts = scipy.spatial.distance.cdist(dev,all, 'euclidean').T
                min_dist = np.argmin(dsts, axis=1)
                row = np.argmax(np.min(dsts, axis=1))
                col = min_dist[row]

                item = all[row, :]

            if (mode == "min"):
                all = r.vector.values
                if (len(all) > 1): all = np.row_stack(all)
                else: all = [all[0]]

                dsts = scipy.spatial.distance.cdist(dev, all, 'euclidean')
                dev_indx, all_indx = np.unravel_index(dsts.argmin(), dsts.shape)
                item = all[all_indx, :]

            dsts = scipy.spatial.distance.cdist(dev, [item], 'euclidean')
            min_dst = np.min(dsts)

            return np.min(dsts)

        # Recomendación de imágenes --------------------------------------------------------------------------------
        test_img_res = pd.DataFrame()

        for batch_di in np.array_split(self.TEST_V3, 10):
            batch_dtfm_imgs_ret = batch_di.copy()
            batch_dtfm_imgs = batch_di[['id_user', 'id_restaurant', 'vector', 'like']]
            batch_dm = batch_dtfm_imgs.values

            feed_dict = {"user_rest_input:0": batch_dm[:, [0, 1]],
                         "img_input:0": np.row_stack(batch_dm[:, [2]][:, 0]),
                         "bin_labels:0": batch_dm[:, [3]],
                         'dpout:0': 1.0}

            batch_bin_prob, batch_softplus = self.SESSION.run(['batch_bin_prob:0', 'batch_softplus:0'], feed_dict=feed_dict)
            batch_dtfm_imgs_ret['prediction'] = batch_bin_prob[:, 0]

            test_img_res = test_img_res.append(batch_dtfm_imgs_ret, ignore_index=True)

        for i, r in test_img_res.groupby("reviewId"):
            r = r.sort_values("prediction")

            best = r.iloc[-1, :]['vector']
            rndm = r.sample(1)['vector'].values[0]

            dev = r.loc[r.is_dev == 1, 'vector'].values

            min_dst = getDist(dev, r, mode="best")
            min_dst_rnd = getDist(dev, r, mode="random")
            min_dst_cnt = getDist(dev, r, mode="centroid")
            min_dst_max = getDist(dev, r, mode="max")
            min_dst_min = getDist(dev, r, mode="min")

            dev_img_loss.append(min_dst)
            dev_img_loss_rndm.append(min_dst_rnd)
            dev_img_loss_cnt.append(min_dst_cnt)
            dev_img_loss_max.append(min_dst_max)
            dev_img_loss_min.append(min_dst_min)

        return (dev_img_loss,dev_img_loss_rndm,dev_img_loss_cnt,dev_img_loss_max,dev_img_loss_min)

    def TrainPrint(self,epoch, train_ret, test_ret):

        if(len(test_ret)==0):print(train_ret[1]);return()

        header = ["E","ACC_T","MIN_D", "RNDM", "CNTR", "MAX  ", "MIN  "]
        header_line = "\t".join(header)
        print(header_line)

        log_items = [epoch + 1]
        log_items.append(train_ret[1])

        log_items.append(bcolors.OKBLUE)
        log_items.extend(np.round([np.mean(test_ret[0]),np.mean(test_ret[1]),np.mean(test_ret[2]),np.mean(test_ret[3]),np.mean(test_ret[4])], decimals=4))
        log_items.append(bcolors.ENDC)

        log_line = "\t".join(map(lambda x: str(x), log_items))
        log_line = log_line.replace("\t\t", "\t")
        print(log_line)

    def gridSearchPrint(self,epoch,train,dev):

        log_items = [epoch + 1]

        log_items.extend(np.round([train[0], dev[0]], decimals=4))
        log_items.append(bcolors.OKBLUE)
        log_items.extend(np.round([train[1],dev[1]], decimals=4))
        log_items.append(bcolors.ENDC)

        log_line = "\t".join(map(lambda x: str(x), log_items))
        log_line = log_line.replace("\t\t", "\t")

        print(log_line)

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()
