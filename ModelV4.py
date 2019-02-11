# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV4(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv4  (IMG-PREFERENCES)"
        ModelClass.__init__(self,city,option,config,seed = seed, name = modelName)

    def getModel(self):

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']

            # Número global de iteraciones
            global_step_bin = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_bin')

            # Datos de entrada -----------------------------------------------------------------------------------------------------------

            user_rest_input = tf.placeholder(tf.int32, shape=[None, 2], name="user_rest_input")
            img_input_best = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input_best")
            img_input_worst  = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input_worst")

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E2")
            E3 = tf.Variable(tf.truncated_normal([self.V_IMG, emb_size*2], mean=0.0, stddev=1.0 / math.sqrt(emb_size*2)),name="E3")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_rest_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, user_rest_input[:,1])

            img_emb_best = tf.matmul(img_input_best, E3, name='img_emb_best')
            img_emb_worst = tf.matmul(img_input_worst, E3, name='img_emb_worst')

            user_rest_emb = tf.concat([user_emb,rest_emb],axis=1, name="user_rest_emb")

            # Cálculo de LOSS y optimizador ----------------------------------------------------------------------------------------------

            dot_best = tf.reduce_sum(tf.multiply(user_rest_emb, img_emb_best), 1, keepdims=True, name="dot_prod")
            dot_worst = tf.reduce_sum(tf.multiply(user_rest_emb, img_emb_worst), 1, keepdims=True, name="dot_prod")

            batch_loss = tf.math.maximum(0.0,1-(dot_best-dot_worst), name="batch_loss")
            loss = tf.reduce_sum(batch_loss)

            train_step_bin = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate'])\
                .minimize(loss=loss, global_step=global_step_bin)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph

    def train(self):

        tss = time.time()

        train_bin_loss = []
        train_bin_batches = np.array_split(self.TRAIN_V3, len(self.TRAIN_V3) // self.CONFIG['batch_size'])

        for bn in range(len(train_bin_batches)):

            batch_dtfm_imgs_ret = train_bin_batches[bn][['id_user', 'id_restaurant','vector','like']]
            batch_train_bin = batch_dtfm_imgs_ret.values

            feed_dict_bin = {
                "user_rest_input:0": batch_train_bin[:, [0, 1]],
                "img_input:0": np.row_stack(batch_train_bin[:, [2]][:, 0]),
            }

            _, batch_loss,dot_prod = self.SESSION.run(['train_step_bin:0', 'batch_loss:0','dot_prod:0'], feed_dict=feed_dict_bin)

            train_bin_loss.extend(batch_loss[:, 0])


        return (np.mean(train_bin_loss))

    def dev(self):

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

        # Recomendación de restaurantes --------------------------------------------------------------------------------

        dev_bin_res = pd.DataFrame()

        dev_loss = []
        dev_img_loss = []
        dev_img_loss_rndm = []
        dev_img_loss_max = []
        dev_img_loss_min = []
        dev_img_loss_cnt = []

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

        hits, avg_pos, median_pos = self.getTopN(dev_bin_res, data=self.DEV)
        hits = list(hits.values())

        # Si se entrena con las imágenes
        if (self.CONFIG['use_images'] == 1):

            # Recomendación de imágenes --------------------------------------------------------------------------------
            dev_img_res = pd.DataFrame()

            for batch_di in np.array_split(self.DEV_V3, 10):
                batch_dtfm_imgs_ret = batch_di.copy()
                batch_dtfm_imgs = batch_di[['id_user', 'id_restaurant','vector','like']]
                batch_dm = batch_dtfm_imgs.values

                feed_dict = {"user_rest_input:0": batch_dm[:, [0, 1]],
                                 "img_input:0": np.row_stack(batch_dm[:, [2]][:, 0]),
                                 "bin_labels:0": batch_dm[:, [3]],
                                 'dpout:0': 1.0}

                batch_bin_prob, batch_softplus = self.SESSION.run(['batch_bin_prob:0', 'batch_softplus:0'], feed_dict=feed_dict)
                batch_dtfm_imgs_ret['prediction'] = batch_bin_prob[:, 0]

                dev_img_res = dev_img_res.append(batch_dtfm_imgs_ret, ignore_index=True)

            for i,r in dev_img_res.groupby("reviewId"):
                r = r.sort_values("prediction")

                best = r.iloc[-1,:]['vector']
                rndm = r.sample(1)['vector'].values[0]

                dev = r.loc[r.is_dev==1,'vector'].values

                min_dst = getDist(dev,r,mode = "best")

                if(self.GS_EPOCH==0):
                    min_dst_rnd = getDist(dev,r,mode = "random")
                    min_dst_cnt = getDist(dev,r,mode = "centroid")
                    min_dst_max = getDist(dev,r,mode = "max")
                    min_dst_min = getDist(dev,r,mode = "min")

                else: min_dst_rnd = min_dst_cnt = min_dst_max = min_dst_min = -1

                dev_img_loss.append(min_dst)
                dev_img_loss_rndm.append(min_dst_rnd)
                dev_img_loss_cnt.append(min_dst_cnt)
                dev_img_loss_max.append(min_dst_max)
                dev_img_loss_min.append(min_dst_min)

        return ((np.average(dev_loss), hits, avg_pos, median_pos,dev_img_loss,dev_img_loss_rndm,dev_img_loss_cnt,dev_img_loss_max,dev_img_loss_min), avg_pos)

    def gridSearchPrint(self,epoch,train,dev):

        if(epoch==0):
            header = ["E","T_LOSS","D_LOSS","","TOP_1","TOP_5","TOP_10","TOP_15","TOP_20","","MEAN","MEDIAN"]
            if (self.CONFIG['use_images'] == 1): header.extend(["MIN_D","RNDM","CNTR","MAX  ","MIN  "])
            header_line = "\t".join(header)
            print(header_line)

        log_items = [epoch+1]

        log_items.extend(np.round([train, dev[0]], decimals=4))
        log_items.append(bcolors.OKBLUE)
        log_items.extend(np.round(dev[1], decimals=4))
        log_items.append(bcolors.ENDC)

        log_items.extend(np.round([dev[2], dev[3]], decimals=4))

        # Si se entrena con las imágenes
        if (self.CONFIG['use_images'] == 1): log_items.extend(np.round([np.mean(dev[4]),np.mean(dev[5]),np.mean(dev[6]),np.mean(dev[7]),np.mean(dev[8])],decimals=4))

        log_line = "\t".join(map(lambda x: str(x), log_items))
        log_line = log_line.replace("\t\t", "\t")

        print(log_line)

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()
