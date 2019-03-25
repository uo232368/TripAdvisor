# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV4(ModelClass):

    def __init__(self,city,option,config,date,seed = 2,modelName= "modelv4"):

        ModelClass.__init__(self,city,option,config,modelName,date,seed = seed)
        self.IMG = np.row_stack(self.IMG.vector.values)

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
            E3 = tf.Variable(tf.truncated_normal([self.V_IMG, emb_size*2], mean=0.0, stddev=1.0 / math.sqrt(emb_size*2)),name="E3")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, rest_input[:,0])

            rest_worst_emb = tf.nn.embedding_lookup(E2, rest_input_worst[:,0])

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

    def getBaselines(self, test=False):

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
        if(test==False):TRAIN = self.getPickle(path, "TRAIN")
        else:TRAIN = self.getPickle(path, "TRAIN_DEV")

        for i, g in TRAIN.groupby("id_restaurant"):
            all_c = self.IMG[g.id_img.values, :]
            cnt = np.mean(all_c, axis=0)

            RST_CNTS = RST_CNTS.append({"id_restaurant": i, "vector": cnt}, ignore_index=True)

        # Para cada caso de DEV, calcular el resultado de los baselines
        # ---------------------------------------------------------------------------------------------------------------

        RET = pd.DataFrame(columns=["id_user","id_restaurant","n_photos","n_photos_dev","cnt_pos","rnd_pos"])

        if (test == False):ITEMS = self.DEV
        else:ITEMS = self.TEST

        for i, g in ITEMS.groupby("id_user"):

            cnt_pos = getPos(g,RST_CNTS)
            rnd_pos = getRndPos(g)

            RET = RET.append({"id_user":i,"id_restaurant":g.id_restaurant.values[0],"n_photos":len(g),"n_photos_dev":len(g.loc[g.is_dev==1]),"cnt_pos":cnt_pos,"rnd_pos":rnd_pos},ignore_index=True)


        RET["PCNT_CNT"] = RET.apply(lambda x: x.cnt_pos/x.n_photos , axis=1)
        RET["PCNT-1_CNT"] = RET.apply(lambda x: (x.cnt_pos-1)/x.n_photos , axis=1)

        RET["PCNT_RND"] = RET.apply(lambda x: x.rnd_pos/x.n_photos , axis=1)
        RET["PCNT-1_RND"] = RET.apply(lambda x: (x.rnd_pos-1)/x.n_photos , axis=1)

        RET.to_excel("docs/"+self.DATE+"/BaselineModels"+self.CITY+("_TEST" if test else "_DEV")+".xls")

        print("PCNT_CNT:\t",RET["PCNT_CNT"].mean())
        print("PCNT-1_CNT:\t",RET["PCNT-1_CNT"].mean())
        print("PCNT_RND:\t",RET["PCNT_RND"].mean())
        print("PCNT-1_RND:\t",RET["PCNT-1_RND"].mean())

        return RET

    def train(self):

        train_df = pd.DataFrame()

        train_bin_loss = []
        train_bin_batches = np.array_split(self.TRAIN, len(self.TRAIN) // self.CONFIG['batch_size'])

        for bn in range(len(train_bin_batches)):

            batch_dtfm_imgs_ret = train_bin_batches[bn][['id_user', 'id_restaurant','id_img','id_rest_worst','worst']]
            ret = batch_dtfm_imgs_ret.copy()
            batch_train_bin = batch_dtfm_imgs_ret.values

            feed_dict_bin = {
                "dropout:0": self.CONFIG["dropout"],
                "is_train:0":True,
                "user_input:0": batch_train_bin[:, [0]],
                "rest_input:0": batch_train_bin[:, [1]],
                "img_input_best:0": self.IMG[batch_train_bin[:, [2]][:,0],:],
                "rest_input_worst:0":  batch_train_bin[:, [3]],
                "img_input_worst:0": self.IMG[batch_train_bin[:, [4]][:,0],:]
            }

            _, gs,lr,batch_loss,dot_best, dot_worst = self.SESSION.run(['train_step_bin:0','global_step_bin:0','learning_rate:0', 'batch_loss:0','dot_best:0','dot_worst:0'], feed_dict=feed_dict_bin)

            train_bin_loss.extend(batch_loss[:, 0])
            #ret['best'] = dot_best[:, 0]
            #ret['worst'] = dot_worst[:, 0]
            #ret['loss'] = batch_loss[:, 0]

            #train_df = train_df.append(ret)

        #print(lr)

        return (np.mean(train_bin_loss), lr)

    def dev(self):

        def getPos(r):

            r = r.reset_index(drop=True)
            id_r = r.id_restaurant.unique()[0]
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            return pos

        # Recomendación de restaurantes --------------------------------------------------------------------------------

        dev_bin_res = pd.DataFrame()

        dev_loss = []

        pos_model = []
        pcnt_model = []
        pcnt1_model = []

        # Recomendación de imágenes --------------------------------------------------------------------------------
        dev_img_res = pd.DataFrame()

        for batch_di in np.array_split(self.DEV, 10):
            batch_dtfm_imgs_ret = batch_di.copy()
            batch_dtfm_imgs = batch_di[['id_user', 'id_restaurant','id_img']]
            batch_dm = batch_dtfm_imgs.values

            feed_dict_bin = {
                "dropout:0":1,
                "is_train:0": False,
                "user_input:0": batch_dm[:, [0]],
                "rest_input:0": batch_dm[:, [1]],
                "img_input_best:0": self.IMG[batch_dm[:, [2]][:,0],:]
            }

            dot_best = self.SESSION.run('dot_best:0', feed_dict=feed_dict_bin)

            batch_dtfm_imgs_ret['prediction'] = dot_best[:, 0]
            dev_img_res = dev_img_res.append(batch_dtfm_imgs_ret, ignore_index=True)

        RET = pd.DataFrame(columns=["id_user","id_restaurant","n_photos","n_photos_dev","model_pos","pcnt_model","pcnt1_model"])

        for i,r in dev_img_res.groupby(["id_user","id_restaurant"]):
            r = r.sort_values("prediction", ascending=False)
            dev = r.loc[r.is_dev==1,'id_img'].values

            item_pos = getPos(r)

            pos_model.append(item_pos)
            pcnt_model.append(item_pos/len(r))
            pcnt1_model.append((item_pos-1)/len(r))

            RET = RET.append({"id_user":i[0],"id_restaurant":i[1],"n_photos":len(r),"n_photos_dev":len(dev),"model_pos":pos_model[-1],"pcnt_model":pcnt_model[-1],"pcnt1_model":pcnt1_model[-1]},ignore_index=True)

        if(self.CURRENT_EPOCH==(self.CONFIG["epochs"]-1) or self.CURRENT_EPOCH==-1):

            RET.to_excel("docs/"+self.DATE+"/"+self.MODEL_NAME+"_"+self.CITY+"_byRestaurant.xls") # Por restaurante
            RET = RET[['id_user', 'id_restaurant', 'n_photos_dev', 'model_pos','pcnt_model', 'pcnt1_model']]

            #ESTO SOLO VALE PARA TEST!!!!!!!!!!!!!!!!!!!!!!!!!!

            items = [9,5,4,2,1]

            TRAIN_DATA = self.getPickle(self.DATA_PATH + "original/", "TRAIN_DEV")
            TRAIN_DATA = TRAIN_DATA.groupby("id_user").apply(lambda x : pd.Series({"n_photos_train":len(x),"n_rest_train":len(x.id_restaurant.unique())})).reset_index()
            RET = TRAIN_DATA.merge(RET, right_on="id_user", left_on="id_user")

            BASELINE = pd.read_excel("docs/" + self.DATE + "/BaselineModels" + self.CITY + "_TEST.xls")[["id_user", "PCNT-1_CNT", "PCNT-1_RND"]]
            RET = BASELINE.merge(RET, right_on="id_user", left_on="id_user")

            RET["RND-MOD"] = RET["PCNT-1_RND"]-RET["pcnt1_model"]
            RET["CNT-MOD"] = RET["PCNT-1_CNT"]-RET["pcnt1_model"]

            RET = RET.sort_values("n_photos_train",ascending=False).reset_index(drop=True)

            def getAcAverage(data):
                ret = []
                for i, v in enumerate(data):
                    ret.append(np.mean(data[:i+1]))
                return ret

            RET["RND-MOD_AC"] = getAcAverage(RET["RND-MOD"].values)
            RET["CNT-MOD_AC"] = getAcAverage(RET["CNT-MOD"].values)

            print("-"*100)
            print("N_FOTOS_TRAIN (>=)\tN_ITEMS\t%ITEMS\tRND-MOD AC\tCNT-MOD AC\tMODELO")
            for it in items:
                data = RET.loc[RET["n_photos_train"]>=it]
                print(str(it) + "\t" + str(len(data))+ "\t" +str(len(data)/len(RET)) + "\t" + str(data.iloc[-1, data.columns.get_loc("RND-MOD_AC")]) + "\t" + str(data.iloc[-1, data.columns.get_loc("CNT-MOD_AC")]) + "\t" + str(data["pcnt1_model"].mean()))
            print("-"*100)

            #RET.to_excel("docs/"+self.DATE+"/"+self.MODEL_NAME+"_"+self.CITY+"_byUser.xls") # Por usuario

        return ((pos_model, pcnt_model, pcnt1_model), np.mean(pcnt1_model))

    def gridSearchPrint(self,epoch,train,dev):

        if(epoch==0):
            header = ["E","LRN_RATE","T_LOSS","MEAN_POS","PCNT","PCNT-1"]
            header_line = "\t".join(header)
            print(header_line)

        log_items = [epoch+1]
        log_items.append(np.round(train[1], decimals=7))
        log_items.extend(np.round([train[0]], decimals=4))
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

                TRAIN_PREFS = utils.shuffle(TRAIN_PREFS, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_PREFS", TRAIN_PREFS);
                del TRAIN_PREFS

            else: self.printW("Ya existen las preferencias de TRAIN, se omite...")

            if (not os.path.exists(file_path + "TRAIN_DEV_PREFS")):

                TRAIN_DEV_PREFS = self.getPickle(split_file_path, "TRAIN_DEV")
                #TRAIN_DEV_PREFS["item"] = TRAIN_DEV_PREFS.index

                #TRAIN_DEV_PREFS = TRAIN_DEV_PREFS.groupby("item").apply(trainFn, TRAIN_DEV_PREFS).reset_index(drop=True)
                TRAIN_DEV_PREFS = getPreferences(TRAIN_DEV_PREFS)
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
                self.toPickle(file_path, "DEV", DEV); del DEV

            else: self.printW("Ya se han añadido los restaurantes a DEV, se omite...")

            if(not os.path.exists(file_path+"TEST")):
                TEST = self.getPickle(split_file_path, "TEST")
                TRAIN_DEV = self.getPickle(split_file_path, "TRAIN_DEV")

                TEST = TEST.groupby(["id_user", "id_restaurant"]).apply(testFn, TRAIN_DEV).reset_index(drop=True)
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
