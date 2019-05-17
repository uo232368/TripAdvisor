# -*- coding: utf-8 -*-
from src.ModelClass import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

########################################################################################################################

class ModelV5(ModelClass):

    def __init__(self,city,option,config,date,seed = 2,modelName= "modelv5"):
        ModelClass.__init__(self,city,option,config,modelName,date,seed = seed)

    def getModel(self):

        def load_image(path):
            image = tf.read_file(self.IMG_PATH+ path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize_images(image, [self.CONFIG["img_size"], self.CONFIG["img_size"]])
            image /= 255.0  # normalize to [0,1] range
            return image

        img_width = self.CONFIG["img_size"]
        img_height = img_width

        # Creación del grafo de TF.
        graph = tf.Graph()

        with graph.as_default():

            tf.set_random_seed(self.SEED)

            emb_size = self.CONFIG['emb_size']
            usr_emb_size=emb_size #512
            rst_emb_size=emb_size//2   #256
            img_emb_size=emb_size*2   #1024

            concat_size = usr_emb_size + rst_emb_size + img_emb_size

            # Variables
            global_step_bin = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step_bin')

            # Datos de entrada -----------------------------------------------------------------------------------------------------------

            is_train = tf.placeholder(tf.bool, name="is_train");
            dropout = tf.placeholder(tf.float32, name="dropout");
            batch_size = tf.placeholder(tf.int64, name="batch_size")

            user_input = tf.placeholder(tf.int32, shape=[None], name="user_input")
            rest_input = tf.placeholder(tf.int32, shape=[None], name="rest_input")
            img_input_best = tf.placeholder(tf.string, shape=[None], name="img_input_best")
            rest_input_worst = tf.placeholder(tf.int32, shape=[None], name="rest_input_worst")
            img_input_worst = tf.placeholder(tf.string, shape=[None], name="img_input_worst")

            # Crear los datasets correspondientes --------------------------------------------------------------------------

            user_ds = tf.data.Dataset.from_tensor_slices(user_input)
            rest_ds = tf.data.Dataset.from_tensor_slices(rest_input)
            img_best_ds = tf.data.Dataset.from_tensor_slices(img_input_best)
            img_best_ds = img_best_ds.map(load_image, num_parallel_calls=5)
            rest_worst_ds = tf.data.Dataset.from_tensor_slices(rest_input_worst)
            img_worst_ds = tf.data.Dataset.from_tensor_slices(img_input_worst)
            img_worst_ds = img_worst_ds.map(load_image, num_parallel_calls=5)

            #dataset = tf.data.Dataset.zip((user_ds, rest_ds, img_best_ds, rest_worst_ds, img_worst_ds)).batch(batch_size)
            #dataset = dataset.prefetch(buffer_size=-1)

            B = tf.data.Dataset.zip((user_ds, rest_ds, img_best_ds))
            B = B.batch(batch_size)
            B = B.prefetch(buffer_size=-1)

            W = tf.data.Dataset.zip((rest_worst_ds, img_worst_ds))
            W = W.batch(batch_size)
            W = W.prefetch(buffer_size=-1)

            # Se crea un iterador indicando el tipo de datos de entrada y el tamaño
            iter_b = tf.data.Iterator.from_structure(B.output_types, B.output_shapes)
            iter_w = tf.data.Iterator.from_structure(W.output_types, W.output_shapes)

            user_batch, rest_batch, img_batch_best = iter_b.get_next()
            rest_batch_worst, img_batch_worst = iter_w.get_next()

            #user_batch, rest_batch, img_batch_best, rest_batch_worst, img_batch_worst = iter.get_next()  # Pedir un batch

            init_iter_b = iter_b.make_initializer(B, name="init_iter_b")  # Operación de inicializar datos
            init_iter_w = iter_w.make_initializer(W, name="init_iter_w")  # Operación de inicializar datos


            def imgNet(input):
                conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv1")
                conv2 = tf.layers.conv2d(inputs=conv1, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv2")
                conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv3")
                conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv4")

                pool1 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2 )

                conv5 = tf.layers.conv2d(inputs=pool1, filters=64,  kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv5")
                conv6 = tf.layers.conv2d(inputs=conv5, filters=64,  kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv6")
                conv7 = tf.layers.conv2d(inputs=conv6, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE,  name="conv7")
                conv8 = tf.layers.conv2d(inputs=conv7, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE,  name="conv8")
                conv9 = tf.layers.conv2d(inputs=conv8, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE,  name="conv9")
                conv10 = tf.layers.conv2d(inputs=conv9, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv10")

                pool2 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2 )

                conv11 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv11")
                conv12 = tf.layers.conv2d(inputs=conv11, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv12")
                conv13 = tf.layers.conv2d(inputs=conv12, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv13")
                conv14 = tf.layers.conv2d(inputs=conv13, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,reuse=tf.AUTO_REUSE, name="conv14")

                flat1 = tf.layers.flatten(inputs=conv14, name="flat1")
                dense1 = tf.layers.dense(inputs=flat1, units=img_emb_size*2, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="dense1")
                dop1 = tf.layers.dropout(inputs=dense1, rate=dropout,  name="dop1")
                #dense2 = tf.layers.dense(inputs=dop1, units=emb_size, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="dense2")
                #dop2 = tf.layers.dropout(inputs=dense2, rate=dropout,  name="dop2")
                out = tf.layers.dense(inputs=dop1, units=img_emb_size, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="dense3")
                #out = tf.reduce_mean(conv6, axis=[1, 2])
                return out

            img_emb_best = imgNet(img_batch_best)
            img_emb_worst = imgNet(img_batch_worst)

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, usr_emb_size], mean=0.0, stddev=1.0 / math.sqrt(usr_emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, rst_emb_size], mean=0.0, stddev=1.0 / math.sqrt(rst_emb_size)),name="E2")

            T0 = tf.Variable(tf.truncated_normal([concat_size,emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="T0")
            T1 = tf.Variable(tf.truncated_normal([emb_size,emb_size//2], mean=0.0, stddev=1.0 / math.sqrt(emb_size//2)),name="T1")
            T2 = tf.Variable(tf.truncated_normal([emb_size//2,emb_size//4], mean=0.0, stddev=1.0 / math.sqrt(emb_size//4)),name="T2")
            T3 = tf.Variable(tf.truncated_normal([emb_size//4,1], mean=0.0, stddev=1.0 / math.sqrt(1)),name="T3")

            bst0 = tf.Variable(tf.zeros(emb_size),name="bt0")
            bst1 = tf.Variable(tf.zeros(emb_size//2),name="bt1")
            bst2 = tf.Variable(tf.zeros(emb_size//4),name="bt2")

            # Operaciones entrada --------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_batch)
            rest_emb = tf.nn.embedding_lookup(E2, rest_batch)
            rest_worst_emb = tf.nn.embedding_lookup(E2, rest_batch_worst)

            # Operaciones capas ocultas --------------------------------------------------------------------------------------------

            best_concat = tf.concat([user_emb,rest_emb,img_emb_best],axis=1)
            best_concat = tf.nn.dropout(best_concat, name='best_concat', keep_prob=dropout)

            worst_concat = tf.concat([user_emb,rest_worst_emb,img_emb_worst],axis=1)
            worst_concat = tf.nn.dropout(worst_concat, name='worst_concat', keep_prob=dropout)

            bt0 = tf.nn.dropout(tf.nn.relu(tf.matmul(best_concat, T0)+bst0),keep_prob=dropout, name='bt0')
            bt1 = tf.nn.dropout(tf.nn.relu(tf.matmul(bt0, T1)+bst1),keep_prob=dropout, name='bt1')
            bt2 = tf.nn.dropout(tf.nn.relu(tf.matmul(bt1, T2)+bst2),keep_prob=dropout, name='bt2')
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
            loss = tf.reduce_sum(batch_loss, name="loss")

            decay_steps = (len(self.TRAIN) // self.CONFIG["batch_size"]) * self.CONFIG["epochs"]
            learning_rate = tf.train.linear_cosine_decay(self.CONFIG['learning_rate'], global_step_bin, decay_steps,name="learning_rate")

            adam = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=learning_rate)
            train_step_bin = adam.minimize(loss=loss, global_step=global_step_bin)


        return graph

    def getBaselines(self):

        def getPos(data, ITEMS):
            g = data.copy()
            id_rest = g.id_restaurant.unique()[0]
            item = ITEMS.loc[ITEMS.id_restaurant==id_rest].vector.values[0]

            rst_imgs = self.IMG[g.id_img.values,:]
            dsts = scipy.spatial.distance.cdist(rst_imgs, [item], 'euclidean')

            g["dsts"] = dsts

            g = g.sort_values("dsts").reset_index(drop=True)

            return min(g.loc[g.is_dev==1].index.values)+1

        def getRndPos(data):
            g = data.copy()
            g["prob"] = np.random.random_sample( len(g))
            g = g.sort_values("prob").reset_index(drop=True)

            return len(g)-max(g.loc[g.is_dev == 1].index.values)

        # Obtener los centroides de los restaurantes en train y los random
        #---------------------------------------------------------------------------------------------------------------
        RST_CNTS = pd.DataFrame(columns=["id_restaurant", "vector"])

        # Obtener el conjunto de TRAIN original
        path = self.DATA_PATH+ "original/"
        TRAIN = self.getPickle(path, "TRAIN")

        # ToDo: CUANDO SE ENTRENE EL MODELO FINAL; HAY QUE CALCULAR ESTO SOBRE TRAIN_DEV PARA TEST

        for i, g in TRAIN.groupby("id_restaurant"):
            all_c = self.IMG[g.id_img.values, :]
            cnt = np.mean(all_c, axis=0)

            RST_CNTS = RST_CNTS.append({"id_restaurant": i, "vector": cnt}, ignore_index=True)

        # Para cada caso de DEV, calcular el resultado de los baselines
        # ---------------------------------------------------------------------------------------------------------------

        RET = pd.DataFrame(columns=["id_user","id_restaurant","n_photos","n_photos_dev","cnt_pos","rnd_pos"])

        for i, g in self.DEV.groupby("id_user"):

            cnt_pos = getPos(g,RST_CNTS)
            rnd_pos = getRndPos(g)

            RET = RET.append({"id_user":i,"id_restaurant":g.id_restaurant.values[0],"n_photos":len(g),"n_photos_dev":len(g.loc[g.is_dev==1]),"cnt_pos":cnt_pos,"rnd_pos":rnd_pos},ignore_index=True)


        RET["PCNT_CNT"] = RET.apply(lambda x: x.cnt_pos/x.n_photos , axis=1)
        RET["PCNT-1_CNT"] = RET.apply(lambda x: (x.cnt_pos-1)/x.n_photos , axis=1)

        RET["PCNT_RND"] = RET.apply(lambda x: x.rnd_pos/x.n_photos , axis=1)
        RET["PCNT-1_RND"] = RET.apply(lambda x: (x.rnd_pos-1)/x.n_photos , axis=1)

        RET.to_excel("docs/"+self.DATE+"/BaselineModels"+self.CITY+".xls")

        print("PCNT_CNT:\t",RET["PCNT_CNT"].mean())
        print("PCNT-1_CNT:\t",RET["PCNT-1_CNT"].mean())
        print("PCNT_RND:\t",RET["PCNT_RND"].mean())
        print("PCNT-1_RND:\t",RET["PCNT-1_RND"].mean())

        return RET

    def train(self):

        if(len(self.TRAIN) % self.CONFIG["batch_size"]==0):
            n_batches = (len(self.TRAIN) // self.CONFIG["batch_size"])
        else:
            n_batches = (len(self.TRAIN) // self.CONFIG["batch_size"]) +1

        train_loss = []

        # Cargar los datos en el iterador
        self.SESSION.run(["init_iter_b","init_iter_w"], feed_dict={
            "batch_size:0": self.CONFIG["batch_size"],
            "user_input:0": self.TRAIN.id_user.values,
            "rest_input:0": self.TRAIN.id_restaurant.values,
            "img_input_best:0": self.TRAIN.best_path.values,
            "rest_input_worst:0": self.TRAIN.id_rest_worst.values,
            "img_input_worst:0": self.TRAIN.worst_path.values})

        for b in range(n_batches):
            _, batch_loss = self.SESSION.run(["train_step_bin", "batch_loss:0"], feed_dict={"is_train:0":True,"dropout:0":self.CONFIG["dropout"]})
            train_loss.extend(batch_loss[:, 0])

        return (np.mean(train_loss))

    def dev(self):

        def getPos(r):

            r = r.reset_index(drop=True)
            id_r = r.id_restaurant.unique()[0]
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            return pos

        if(len(self.DEV) % self.CONFIG["batch_size"]==0):
            n_batches = (len(self.DEV) // self.CONFIG["batch_size"])
        else:
            n_batches = (len(self.DEV) // self.CONFIG["batch_size"]) +1

        dev_pred = []

        pos_model = []
        pcnt_model = []
        pcnt1_model = []

        # Cargar los datos en el iterador
        self.SESSION.run("init_iter_b", feed_dict={
            "batch_size:0": self.CONFIG["batch_size"],
            "user_input:0": self.DEV.id_user.values,
            "rest_input:0": self.DEV.id_restaurant.values,
            "img_input_best:0": self.DEV.best_path.values})

        for b in range(n_batches):
            dot_best = self.SESSION.run('dot_best:0', feed_dict={"is_train:0":False,"dropout:0":1.0})
            dev_pred.extend(dot_best[:, 0])

        dev_img_res = self.DEV.copy()
        dev_img_res["prediction"] = dev_pred

        RET = pd.DataFrame(columns=["id_user","id_restaurant","n_photos","n_photos_dev","model_pos","pcnt_model","pcnt1_model"])

        for i,r in dev_img_res.groupby(["id_user","id_restaurant"]):
            r = r.sort_values("prediction", ascending=False)
            dev = r.loc[r.is_dev==1]

            item_pos = getPos(r)

            pos_model.append(item_pos)
            pcnt_model.append(item_pos/len(r))
            pcnt1_model.append((item_pos-1)/len(r))

            RET = RET.append({"id_user":i[0],"id_restaurant":i[1],"n_photos":len(r),"n_photos_dev":len(dev),"model_pos":pos_model[-1],"pcnt_model":pcnt_model[-1],"pcnt1_model":pcnt1_model[-1]},ignore_index=True)

        #if(self.CURRENT_EPOCH==10):RET.to_excel("docs/"+self.DATE+"/"+self.MODEL_NAME+"_Results("+self.CONFIG["neg_examples"]+").xls")

        return ((pos_model, pcnt_model, pcnt1_model), np.mean(pcnt1_model))

    def gridSearchPrint(self,epoch,train,dev):

        if(epoch==0):
            header = ["E","T_LOSS","MEAN_POS","PCNT","PCNT-1"]
            header_line = "\t".join(header)
            print(header_line)

        log_items = [epoch+1]
        log_items.extend(np.round([train], decimals=4))
        log_items.extend(np.round([np.mean(dev[0]),np.mean(dev[1]),np.mean(dev[2])],decimals=4))

        log_line = "\t".join(map(lambda x: str(x), log_items))
        log_line = log_line.replace("\t\t", "\t")

        print(log_line)

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()

    ####################################################################################################################

    def getFilteredData(self,verbose=True):

        IMG = pd.read_pickle(self.PATH + "img-option" + str(self.OPTION) + "-new.pkl")
        RVW = pd.read_pickle(self.PATH + "reviews.pkl")

        IMG['review'] = IMG.review.astype(int)
        RVW["reviewId"] = RVW.reviewId.astype(int)

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW = RVW.loc[(RVW.num_images>0)] #Eliminar reviews sin imagen
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW = RVW.loc[(RVW.userId != "")]

        # Obtener ID para ONE-HOT de usuarios y restaurantes
        # ---------------------------------------------------------------------------------------------------------------

        USR_TMP = pd.DataFrame(columns=["real_id", "id_user"])
        REST_TMP = pd.DataFrame(columns=["real_id", "id_restaurant"])

        # Obtener tabla real_id -> id para usuarios
        USR_TMP.real_id = RVW.sort_values("userId").userId.unique()
        USR_TMP.id_user = range(0, len(USR_TMP))

        # Obtener tabla real_id -> id para restaurantes
        REST_TMP.real_id = RVW.sort_values("restaurantId").restaurantId.unique()
        REST_TMP.id_restaurant = range(0, len(REST_TMP))

        # Mezclar datos
        RET = RVW.merge(USR_TMP, left_on='userId', right_on='real_id', how='inner')
        RET = RET.merge(REST_TMP, left_on='restaurantId', right_on='real_id', how='inner')

        RVW = RET[['date', 'images', 'language', 'rating', 'restaurantId', 'reviewId', 'text', 'title', 'url', 'userId', 'num_images', 'id_user', 'id_restaurant', 'like']]

        if (verbose):
            self.printW("\tUsuarios: " + str(len(RVW.userId.unique())))
            self.printW("\tRestaurantes: " + str(len(RVW.restaurantId.unique())))


        return RVW, IMG, USR_TMP,REST_TMP

    def getData(self):

        def splitSets(file_temp):

            def split_fn(data):

                def chooseSubset(d):
                    items = len(d)
                    rst = d.id_restaurant.unique()
                    n_rst = len(rst)

                    d["TO_1"] = 0
                    d["TO_2"] = 0

                    if n_rst < 2:
                        d["TO_1"] = 1;
                    else:
                        d.loc[d.id_restaurant.isin(rst[:-1]), "TO_1"] = 1
                        d.loc[d.id_restaurant == rst[-1], "TO_2"] = 1

                    return d

                # Dividir en 2 conjuntos por review N-1
                data = data.groupby('id_user').apply(chooseSubset)

                Subset1 = data.loc[data.TO_1 == 1]
                Subset2 = data.loc[data.TO_2 == 1]

                # Mover restaurantes que solo aparecen en 2 al 1
                S1_RST = Subset1.id_restaurant.unique()

                Subset1 = Subset1.append(Subset2.loc[~Subset2.id_restaurant.isin(S1_RST)], ignore_index=True)
                Subset2 = Subset2.loc[Subset2.id_restaurant.isin(S1_RST)]

                Subset1 = Subset1.drop(columns=["TO_1", "TO_2"])
                Subset2 = Subset2.drop(columns=["TO_1", "TO_2"])

                return Subset1, Subset2

            if (os.path.exists(file_temp) and len(os.listdir(file_temp)) > 0):
                self.printW("Ya existen conjuntos divididos, se omite...")
                return

            os.makedirs(file_temp, exist_ok=True)

            TRAIN_TEST = RVW
            TRAIN_DEV, TEST = split_fn(TRAIN_TEST)
            TRAIN, DEV = split_fn(TRAIN_DEV)

            print("-" * 50)
            print("TRAIN_TEST: ", len(TRAIN_TEST))
            print("TRAIN_DEV: ", len(TRAIN_DEV))

            print("TRAIN: ", len(TRAIN))
            print("DEV: ", len(DEV))
            print("TEST: ", len(TEST))
            print("-" * 50)

            self.toPickle(file_temp, "TRAIN_TEST", TRAIN_TEST);del TRAIN_TEST
            self.toPickle(file_temp, "TRAIN_DEV", TRAIN_DEV);del TRAIN_DEV

            self.toPickle(file_temp, "TRAIN", TRAIN);del TRAIN
            self.toPickle(file_temp, "DEV", DEV);del DEV
            self.toPickle(file_temp, "TEST", TEST);del TEST

            return

        def createPreferences(IMG):

            def getPreferences(data):

                neg = self.CONFIG['neg_examples'].split("+")
                if(len(neg)>1) : neg_rest, neg_other = list(map(int, neg))
                else: neg_rest = int(neg[0]); neg_other = 0

                ret_list=[]

                for i, g in tqdm(data.iterrows(),total=data.shape[0]):
                    id_r = g.id_restaurant
                    id_u = g.id_user

                    img_best = IMG.loc[IMG.id_img == g.id_img, "vector"].values[0]  # Obtener el vector de ese id

                    id_rest_worst = []
                    img_worst = []

                    if (neg_rest > 0):
                        # Obtener imágenes de otros usuarios del mismo restaurante
                        images_idx_rest = data.loc[(data.id_restaurant == id_r) & (data.id_user != id_u), "id_img"].values  # Imagenes de ese restaurante y otros users

                        if(len(images_idx_rest)>0):
                            images = np.row_stack(IMG.loc[IMG.id_img.isin(images_idx_rest), "vector"].values)

                            dists = scipy.spatial.distance.cdist([img_best], images, "euclidean")
                            indx = np.argsort(dists)[0][-neg_rest:]

                            img_worst.extend(images_idx_rest[indx])
                            id_rest_worst.extend([id_r]*len(indx))

                    if(neg_other > 0):
                        # Obtener neg_rest imágenes de otros usuarios del mismo restaurante
                        # Obtener neg_other imágenes de otros usuarios en otros restaurantes

                        images_other = data.loc[(data.id_restaurant != id_r) & (data.id_user != id_u)].sample(neg_other)
                        images_idx_other = images_other.id_img.values

                        img_worst.extend(images_idx_other)
                        id_rest_worst.extend(images_other.id_restaurant.values)

                    ret_vector = list(zip(id_rest_worst,img_worst))
                    ret_vector = list(map(lambda x: [id_u,id_r, g.id_img, x[0],x[1]], ret_vector))
                    ret_list.extend(ret_vector)

                    assert len(ret_vector) <= (neg_other+neg_rest)

                RET = pd.DataFrame(ret_list)

                ret_cols = data.columns.tolist();
                ret_cols.append("id_rest_worst")
                ret_cols.append("worst")

                RET.columns = ret_cols

                return RET

            if(not os.path.exists(file_path+"TRAIN_PREFS")):

                TRAIN_PREFS = self.getPickle(split_file_path, "TRAIN")
                #TRAIN_PREFS["item"] = TRAIN_PREFS.index

                # Al train se añaden ejemplos n del tipo (u,r, foto de u en r, foto de otro u en r) [las n fotos más lejanas de la del usuario]
                #TRAIN_PREFS = TRAIN_PREFS.groupby("item").apply(trainFn, TRAIN_PREFS).reset_index(drop=True)
                TRAIN_PREFS = getPreferences(TRAIN_PREFS)

                TRAIN_PREFS = TRAIN_PREFS.merge(IMG[["id_img","path"]], on="id_img")
                TRAIN_PREFS = TRAIN_PREFS.rename(index=str, columns={"path": "best_path"})
                TRAIN_PREFS = TRAIN_PREFS.merge(IMG[["id_img","path"]], left_on="worst", right_on="id_img")
                TRAIN_PREFS = TRAIN_PREFS.rename(index=str, columns={"path": "worst_path"})

                TRAIN_PREFS = TRAIN_PREFS[["id_user","id_restaurant","id_rest_worst","best_path","worst_path"]]

                TRAIN_PREFS = utils.shuffle(TRAIN_PREFS, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_PREFS", TRAIN_PREFS);
                del TRAIN_PREFS

            else: self.printW("Ya existen las preferencias de TRAIN, se omite...")

            if (not os.path.exists(file_path + "TRAIN_DEV_PREFS")):

                TRAIN_DEV_PREFS = self.getPickle(split_file_path, "TRAIN_DEV")
                #TRAIN_DEV_PREFS["item"] = TRAIN_DEV_PREFS.index

                #TRAIN_DEV_PREFS = TRAIN_DEV_PREFS.groupby("item").apply(trainFn, TRAIN_DEV_PREFS).reset_index(drop=True)
                TRAIN_DEV_PREFS = getPreferences(TRAIN_DEV_PREFS)

                TRAIN_DEV_PREFS = TRAIN_DEV_PREFS.merge(IMG[["id_img", "path"]], on="id_img")
                TRAIN_DEV_PREFS = TRAIN_DEV_PREFS.rename(index=str, columns={"path": "best_path"})
                TRAIN_DEV_PREFS = TRAIN_DEV_PREFS.merge(IMG[["id_img", "path"]], left_on="worst", right_on="id_img")
                TRAIN_DEV_PREFS = TRAIN_DEV_PREFS.rename(index=str, columns={"path": "worst_path"})

                TRAIN_DEV_PREFS = TRAIN_DEV_PREFS[["id_user","id_restaurant","id_rest_worst","best_path","worst_path"]]

                TRAIN_DEV_PREFS = utils.shuffle(TRAIN_DEV_PREFS, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_DEV_PREFS", TRAIN_DEV_PREFS);
                del TRAIN_DEV_PREFS

            else: self.printW("Ya existen las preferencias de TRAIN_DEV, se omite...")

        def appendRestImages():

            def testFn(data, img_set):
                id_r = data.id_restaurant.values[0]
                id_u = data.id_user.values[0]

                tmp = img_set.loc[img_set.id_restaurant == id_r]
                tmp['is_dev'] = 0
                data['is_dev'] = 1

                tmp = tmp.append(data, ignore_index=True)

                tmp['id_user'] = id_u

                tmp = tmp[['id_user', 'id_restaurant', 'id_img', 'is_dev']]

                return tmp

            TRAIN_TEST = self.getPickle(split_file_path, "TRAIN_TEST")

            if(not os.path.exists(file_path+"DEV")):
                DEV = self.getPickle(split_file_path, "DEV")
                TRAIN = self.getPickle(split_file_path, "TRAIN")

                DEV = DEV.groupby(["id_user", "id_restaurant"]).apply(testFn,TRAIN).reset_index(drop=True)

                DEV = DEV.merge(IMG[["id_img", "path"]], on="id_img")
                DEV = DEV.rename(index=str, columns={"path": "best_path"})
                DEV = DEV[["id_user", "id_restaurant","is_dev","best_path"]]

                self.toPickle(file_path, "DEV", DEV); del DEV

            else: self.printW("Ya se han añadido los restaurantes a DEV, se omite...")

            if(not os.path.exists(file_path+"TEST")):
                TEST = self.getPickle(split_file_path, "TEST")
                TRAIN_DEV = self.getPickle(split_file_path, "TRAIN_DEV")

                TEST = TEST.groupby(["id_user", "id_restaurant"]).apply(testFn, TRAIN_DEV).reset_index(drop=True)

                TEST = TEST.merge(IMG[["id_img", "path"]], on="id_img")
                TEST = TEST.rename(index=str, columns={"path": "best_path"})
                TEST = TEST[["id_user", "id_restaurant","is_dev","best_path"]]

                self.toPickle(file_path, "TEST", TEST);del TEST

            else: self.printW("Ya se han añadido los restaurantes a TEST, se omite...")

            return

        ################################################################################################################

        # Mirar si ya existen los datos
        # --------------------------------------------------------------------------------------------------------------

        file_path = self.PATH + self.MODEL_NAME.upper()
        split_file_path = file_path + "/original/"
        file_path += "/data_" + str(self.CONFIG['neg_examples']) + "/"

        RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();
        RVW = RVW.drop(columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'rating', 'language', 'like'])

        IMG["id_img"] = IMG.index
        IMG["path"]= IMG.apply(lambda x: str(x.review) + "/" + str(x.image - 1) + ".jpg", axis=1)
        IMG = IMG[["review","id_img","path", "vector"]]

        #Para DEBUG (Ver url de fotos)---------------------------------------------------------------------------------
        #RVW = RVW.merge(IMG, left_on="reviewId", right_on="review")
        #RVW["url"] = RVW.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)


        if (os.path.exists(file_path) and len(os.listdir(file_path)) == 7):

            self.printW("Cargando datos generados previamente...")

            TRAIN = self.getPickle(file_path, "TRAIN_PREFS")
            TRAIN_DEV = self.getPickle(file_path,"TRAIN_DEV_PREFS")
            DEV = self.getPickle(file_path, "DEV")
            TEST = self.getPickle(file_path, "TEST")

            REST_TMP = self.getPickle(file_path, "REST_TMP")
            USR_TMP = self.getPickle(file_path, "USR_TMP")
            V_IMG = self.getPickle(file_path, "V_IMG")

            return (TRAIN, TRAIN_DEV, DEV, TEST, IMG, REST_TMP, USR_TMP, V_IMG)

        os.makedirs(file_path, exist_ok=True)

        # ---------------------------------------------------------------------------------------------------------------


        # Merge con las fotos, añadir el nombre de la foto y agrupar varias reviews de mismo user a mismo restaurante
        # --------------------------------------------------------------------------------------------------------------
        RVW = RVW.merge(IMG[["review","id_img"]], left_on="reviewId", right_on="review")

        #RVW["url"] = RVW.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)
        #RVW["name"] = RVW.url.apply(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest())
        RVW = RVW.drop(columns=['images', 'review','reviewId','num_images'])


        # Estadisticas
        # ---------------------------------------------------------------------------------------------------------------


        # Número de reviews de cada usuario
        STS0 = RVW.groupby('id_user').apply(lambda x: pd.Series({"reviews": len(x.id_restaurant.unique())})).reset_index()
        self.plothist(STS0, "reviews", title="Número de reviews por usuario", bins=20,save="docs/" + self.DATE + "/"+self.CITY.lower()+"_hist_rvws_pr_usr.jpg")

        # Número de reviews de cada restaurante
        STS1 = RVW.groupby('id_restaurant').apply(lambda x: pd.Series({"reviews": len(x.id_user.unique())})).reset_index()
        self.plothist(STS1,"reviews",title="Número de reviews de cada restaurante", bins=20, save="docs/"+self.DATE+"/"+self.CITY.lower()+"_hist_rvws_pr_rst.jpg")

        # Numero de fotos de cada review
        STS2 = RVW.groupby(['id_restaurant', 'id_user']).apply(lambda x: pd.Series({"fotos": len(x)})).reset_index()
        self.plothist(STS2,"fotos",title="Número de fotos de cada review", bins=10, save="docs/"+self.DATE+"/"+self.CITY.lower()+"_hist_fotos_pr_rvw.jpg")

        # Numero de fotos de cada restaurante
        STS3 = RVW.groupby('id_restaurant').apply(lambda x: pd.Series({"fotos": len(x)})).reset_index()
        self.plothist(STS3,"fotos",title="Número de fotos de cada restaurante", bins=20, save="docs/"+self.DATE+"/"+self.CITY.lower()+"_hist_fotos_pr_rst.jpg")


        # Mover ejemplos positivos a donde corresponde (N, 1, 1)
        # ---------------------------------------------------------------------------------------------------------------

        splitSets(split_file_path)

        #Todo: Estadisticas para cada conjunto

        # Añadir preferencias a TRAIN y GUARDAR
        # --------------------------------------------------------------------------------------------------------------

        createPreferences(IMG)

        # Añadir resto de fotos del restaurante a DEV y TEST
        # --------------------------------------------------------------------------------------------------------------

        appendRestImages()

        # ALMACENAR PICKLE ------------------------------------------------------------------------------------------------

        self.toPickle(file_path, "REST_TMP", len(REST_TMP))
        self.toPickle(file_path, "USR_TMP", len(USR_TMP))
        self.toPickle(file_path, "V_IMG", len(IMG.iloc[0].vector))

        # Cargar datos creados ------------------------------------------------------------------------------------------------

        TRAIN = self.getPickle(file_path,"TRAIN_PREFS")
        TRAIN_DEV = self.getPickle(file_path,"TRAIN_DEV_PREFS")
        DEV = self.getPickle(file_path,"DEV")
        TEST = self.getPickle(file_path,"TEST")

        return (TRAIN, TRAIN_DEV, DEV, TEST, IMG, len(REST_TMP), len(USR_TMP), len(IMG.iloc[0].vector))

    def gridSearch(self, params, max_epochs = 50, start_n_epochs = 5, last_n_epochs = 5):

        def createCombs():

            def flatten(lst):
                return sum(([x] if not isinstance(x, list) else flatten(x)
                            for x in lst), [])

            combs = []
            level = 0
            for v in params.values():
                if (len(combs)==0):
                    combs = v
                else:
                    combs = list(it.product(combs, v))
                level+=1

                if(level>1):
                    for i in range(len(combs)):
                        combs[i] = flatten(combs[i])

            return pd.DataFrame(combs, columns=params.keys())

        def configNet(comb):
            comb.pop("Index")

            for k in comb.keys():
                assert (k in self.CONFIG.keys())
                self.CONFIG[k]=comb[k]

        #-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#-----#

        combs = createCombs()
        self.printW("Existen "+str(len(combs))+" combinaciones posibles")

        for c in combs.itertuples():

            stop_param = []

            c = dict(c._asdict())

            #Configurar la red
            configNet(c)

            #Crear el modelo
            self.MODEL = self.getModel()

            #Imprimir la configuración
            self.printConfig(filter=c.keys())

            #Configurar y crear sesion
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(graph=self.MODEL, config=config) as sess:

                self.SESSION = sess

                n_batches = len(self.TRAIN) // self.CONFIG["batch_size"]
                sess.run(tf.global_variables_initializer())

                for e in range(max_epochs):

                    train_ret = self.train()

                    dev_ret, stop = self.dev()
                    stop_param.append(stop)

                    #Imprimir linea
                    self.gridSearchPrint(e,train_ret,dev_ret)