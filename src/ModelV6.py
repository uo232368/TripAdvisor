# -*- coding: utf-8 -*-
from src.ModelClass import *

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

########################################################################################################################

class ModelV6(ModelClass):

    def __init__(self,city,config,date,seed = 2,modelName= "modelv6"):

        ModelClass.__init__(self,city,2,config,modelName,date,seed = seed)
        self.DATA["IMG"] = np.row_stack(self.DATA["IMG"].vector.values)

    def stop(self):

        if(self.SESSION!=None):
            self.printW("Cerrando sesión de tensorflow...")
            self.SESSION.close()

        if(self.MODEL!=None):
            tf.reset_default_graph()

    ####################################################################################################################

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
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)

            self.SESSION = sess

            #Conjuntos de entrenamiento
            #train_batches = np.array_split(self.TRAIN, len(self.TRAIN) // self.CONFIG['batch_size'])
            #dev_batches = np.array_split(self.DEV, len(self.DEV)  // (self.CONFIG['batch_size']*2))

            keras.backend.get_session().run(tf.global_variables_initializer())

            #mycallback = self.DevCallback(self,dev_batches)
            tb_call = TensorBoard(log_dir='logs', batch_size= self.CONFIG['batch_size'], write_graph=True, write_images=True,update_freq='batch')


            train_sequence_take = self.TrainSequenceTake(self.DATA["TRAIN_IMG"],self.CONFIG['batch_size'], self)
            train_sequence_like = self.TrainSequenceLike(self.DATA["TRAIN_LIKE"],self.CONFIG['batch_size'], self)

            dev_sequence_take   = self.DevSequenceTake(self.DATA["DEV_IMG"], self.CONFIG['batch_size'], self)
            dev_sequence_like   = self.DevSequenceLike(self.DATA["DEV_LIKE"], self.CONFIG['batch_size'], self)


            for e in range(max_epochs):
                tes = time.time()

                self.CURRENT_EPOCH = e

                train_ret = self.train(train_sequence_take, train_sequence_like)
                dev_ret = self.dev(dev_sequence_take,dev_sequence_like)

                self.gridSearchPrint(e,time.time()-tes, train_ret, dev_ret)

    def getImagePos(self, data):

        def getPos(r):
            r = r.reset_index(drop=True)
            id_r = r.id_restaurant.unique()[0]
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            return pos

        pos_model = []
        pcnt_model = []
        pcnt1_model = []


        for i, r in data.groupby(["id_user", "id_restaurant"]):
            r = r.sort_values("prediction", ascending=False)
            item_pos = getPos(r)

            pos_model.append(item_pos)
            pcnt_model.append(item_pos / len(r))
            pcnt1_model.append((item_pos - 1) / len(r))

        return np.mean(pos_model),np.mean(pcnt_model),np.mean(pcnt1_model)

    def getTopN(self, data):

        def getPos(r):
            r = r.reset_index(drop=True)
            id_r = r.id_restaurant.unique()[0]
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            return pos

        ret = pd.DataFrame()
        pos_model = []

        grp = data.groupby(["id_user","id_top"])

        for i, r in grp:
            r = r.sort_values("prediction", ascending=False)

            item_pos = getPos(r)

            pos_model.append(item_pos)

        ret["position"] = pos_model

        for t in self.CONFIG["top"]:
            ret[str(t)] = (ret["position"]<=t).astype(int)

        #ret2 = ret.iloc[:, 1:].sum(axis=0).reset_index()
        #print(((ret2[0] / len(grp)) * 100).values)

        return np.mean(ret.position), np.median(ret.position)

    def gridSearchPrint(self,epoch,time,train,dev):

        def frm(item):
            return'{:6.3f}'.format(item)

        if (self.CONFIG["use_images"] == 1):
            header = ["E","E_TIME","L_LOSS","T_LOSS", "LIKE","TAKE"]
        else:
            header = ["E","E_TIME","L_LOSS"]
            #header.extend(list(map(lambda x : "TOP_"+str(x),self.CONFIG["top"])))
            header.append("LIKE")

        if(epoch==0):print("\t".join(header))

        line = [time]
        line.extend(train)
        line.extend(dev)
        line = list(map(frm,line))
        line.insert(0,str(epoch))

        print("\t".join(line))

        return None

    ####################################################################################################################

    def getFilteredData(self,verbose=True):

        IMG = pd.read_pickle(self.PATH + "img-option" + str(self.OPTION) + "-new.pkl")
        #IMG = pd.read_pickle(self.PATH + "img-food.pkl")
        RVW = pd.read_pickle(self.PATH + "reviews.pkl")

        IMG['review'] = IMG.review.astype(int)
        RVW["reviewId"] = RVW.reviewId.astype(int)

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
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

        def dropMultipleVisits(data):
            #Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            stay = []

            for i, g in tqdm(data.groupby(["id_user", "id_restaurant"]), desc="Droping multiple visits"):
                stay.append(max(g.reviewId.values))

            return data.loc[data.reviewId.isin(stay)]

        def splitIMGSets(file_temp,dataset):

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

                tqdm.pandas(desc="Spliting sets")

                # Dividir en 2 conjuntos por review N-1
                data = data.groupby('id_user').progress_apply(chooseSubset)

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

            TRAIN_TEST = dataset
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

        def splitLIKESets(file_temp,dataset):

            def split_fn(data):

                def chooseSubset(d):
                    items = len(d)
                    rst = d.id_restaurant.unique()
                    n_rst = len(rst)
                    n_likes = len(d.loc[d.like==1])

                    d["TO_1"] = 0
                    d["TO_2"] = 0

                    if n_rst < 2:
                        d["TO_1"] = 1;
                    else:
                        if(n_likes>0):
                            rt2 = d.loc[d.like==1].id_restaurant.values[-1]
                            d.loc[d.id_restaurant != rt2, "TO_1"] = 1
                            d.loc[d.id_restaurant == rt2, "TO_2"] = 1
                        else:
                            d["TO_1"] = 1;


                    return d

                tqdm.pandas(desc="Spliting sets")

                # Dividir en 2 conjuntos por review N-1
                data = data.groupby('id_user').progress_apply(chooseSubset)

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

            TRAIN_TEST = dataset
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

        def createTRAIN_IMG(IMG):

            def generateItems(data):

                neg = self.CONFIG['neg_images'].split("+")
                if(len(neg)>1) : neg_rest, neg_other = list(map(int, neg))
                else: neg_rest = int(neg[0]); neg_other = 0

                #Por defecto, los ejemplos que vienen son: Fotos echas por usuarios que gustan y no gustan,
                #Por tanto, todas son del usuario
                data["take"] = 1

                ret_list = []
                ret_cols = ["id_user","id_restaurant","id_img","take"]

                #Para las fotos del usaurio que le gustan...
                for i, g in tqdm(data.iterrows(),total=data.shape[0], desc="Train img items"):
                    id_r = g.id_restaurant
                    id_u = g.id_user

                    # Añadir las de dentro
                    inside = data.loc[(data.id_restaurant==id_r) & (data.id_user!=0)].copy()
                    if(len(inside)>=neg_rest):inside = inside.sample(neg_rest)
                    inside["reviewId"] = -1
                    inside["take"] = 0
                    inside["id_user"] = id_u


                    # Añadir las de fuera
                    outside = data.loc[(data.id_restaurant != id_r) & (data.id_user != 0)].copy()
                    if (len(outside) >= neg_other): outside = outside.sample(neg_other)
                    outside["reviewId"] = -1
                    outside["take"] = 0
                    outside["id_user"] = id_u

                    all = inside.append(outside, ignore_index=True)

                    #ret_list.append(g[ret_cols].values.tolist())
                    ret_list.extend([g[ret_cols].values.tolist()]*len(all)) #UPSAMPLING
                    ret_list.extend(all[ret_cols].values.tolist())

                RET =  pd.DataFrame(ret_list)
                RET.columns = ret_cols

                #-------------------------------------------------------------------------------------------------------

                T = RET.loc[(RET['take'] == 1)]
                NT = RET.loc[(RET['take'] == 0)]

                print(len(T), "\t", len(NT))

                #-------------------------------------------------------------------------------------------------------

                return RET

            if(not os.path.exists(file_path+"TRAIN_IMG")):

                TRAIN = self.getPickle(split_file_path_take, "TRAIN")

                # Añadir ids de imágenes a las reviews
                TRAIN = TRAIN.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
                TRAIN = TRAIN.drop(columns=['review'])
                
                TRAIN = generateItems(TRAIN)

                TRAIN = utils.shuffle(TRAIN, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_IMG", TRAIN);
                del TRAIN

            else: self.printW("Ya existen las preferencias de TRAIN, se omite...")

            if (not os.path.exists(file_path + "TRAIN_DEV_IMG")):

                TRAIN_DEV = self.getPickle(split_file_path_take, "TRAIN_DEV")

                # Añadir ids de imágenes a las reviews
                TRAIN_DEV = TRAIN_DEV.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
                TRAIN_DEV = TRAIN_DEV.drop(columns=['review'])

                TRAIN_DEV = generateItems(TRAIN_DEV)

                TRAIN_DEV = utils.shuffle(TRAIN_DEV, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_DEV_IMG", TRAIN_DEV);
                del TRAIN_DEV

            else: self.printW("Ya existen las preferencias de TRAIN_DEV, se omite...")

        def createTRAIN_LIKE(N_RST,N_USR):

            def compensateItems(data):

                new_items = []
                columns   = data.columns.values

                for i,g in tqdm(data.groupby("id_user"),desc="Train like items"):

                    #Si todas las reviews que tiene son positivas, añadir nuevos ejemplos
                    if(g.like.unique()[0]==1 or len(g.like.unique())>1 ):
                        rsts = data.loc[data.id_user!=i].sample(ITEMS_PER_USR).id_restaurant.values
                        n_u_itms = list(zip([-1]*ITEMS_PER_USR,[i]*ITEMS_PER_USR,rsts,[0]*ITEMS_PER_USR))

                        #øversampling
                        #ovrsm = g.sample(len(n_u_itms), replace=True).values
                        #new_items.extend(ovrsm)

                        new_items.extend(n_u_itms)


                new_df = pd.DataFrame(new_items)
                new_df.columns = columns

                data = data.append(new_df, ignore_index=True)

                return data

            #-----------------------------------------------------------------------------------------------------------

            if(self.CONFIG["neg_likes"]=="n"):
                #ITEMS_PER_USR =  int((N_RST / N_USR) * 100)
                ITEMS_PER_USR =  int((N_USR / N_RST) * 5)

            else: ITEMS_PER_USR = int(self.CONFIG["neg_likes"])

            if(not os.path.exists(file_path+"TRAIN_LIKE")):

                TRAIN_LIKE = self.getPickle(split_file_path_like, "TRAIN")
                TRAIN_IMG = self.getPickle(split_file_path_take, "TRAIN")

                TRAIN = TRAIN_LIKE.append(TRAIN_IMG, ignore_index=True)

                TRAIN = compensateItems(TRAIN)

                TRAIN = utils.shuffle(TRAIN, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_LIKE", TRAIN);
                del TRAIN

            else: self.printW("Ya existen las preferencias de TRAIN_LIKE, se omite...")

            if (not os.path.exists(file_path + "TRAIN_DEV_LIKE")):

                TRAIN_DEV_LIKE = self.getPickle(split_file_path_like, "TRAIN_DEV")
                TRAIN_DEV_IMG = self.getPickle(split_file_path_take, "TRAIN_DEV")

                TRAIN_DEV = TRAIN_DEV_LIKE.append(TRAIN_DEV_IMG, ignore_index=True)

                TRAIN_DEV = compensateItems(TRAIN_DEV)

                TRAIN_DEV = utils.shuffle(TRAIN_DEV, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_DEV_LIKE", TRAIN_DEV);
                del TRAIN_DEV

            else: self.printW("Ya existen las preferencias de TRAIN_DEV_LIKE, se omite...")

        def createTestImages(IMG):

            def testFn(data, img_set = None):

                id_r = data.id_restaurant.values[0]
                id_u = data.id_user.values[0]

                tmp = img_set.loc[img_set.id_restaurant == id_r]
                tmp['is_dev'] = 0
                data['is_dev'] = 1

                tmp = tmp.append(data, ignore_index=True)

                tmp['id_user'] = id_u

                tmp = tmp[['id_user', 'id_restaurant', 'id_img', 'is_dev']]

                return tmp

            tqdm.pandas(desc="Creating img test sets")

            if(not os.path.exists(file_path+"DEV_IMG")):
                DEV = self.getPickle(split_file_path_take, "DEV")
                DEV = DEV.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
                DEV = DEV.drop(columns=['review'])

                TRAIN = self.getPickle(split_file_path_take, "TRAIN")
                TRAIN = TRAIN.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
                TRAIN = TRAIN.drop(columns=['review'])

                DEV = DEV.groupby(["id_user", "id_restaurant"]).progress_apply(testFn,img_set = TRAIN).reset_index(drop=True)
                self.toPickle(file_path, "DEV_IMG", DEV); del DEV

            else: self.printW("Ya se ha creado el conjunto de DEV de imágenes, se omite...")

            if(not os.path.exists(file_path+"TEST_IMG")):
                TEST = self.getPickle(split_file_path_take, "TEST")
                TEST = TEST.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
                TEST = TEST.drop(columns=['review'])

                TRAIN_DEV = self.getPickle(split_file_path_take, "TRAIN_DEV")
                TRAIN_DEV = TRAIN_DEV.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
                TRAIN_DEV = TRAIN_DEV.drop(columns=['review'])

                TEST = TEST.groupby(["id_user", "id_restaurant"]).progress_apply(testFn,img_set = TRAIN_DEV).reset_index(drop=True)
                self.toPickle(file_path, "TEST_IMG", TEST);del TEST

            else: self.printW("Ya se ha creado el conjunto de TEST de imágenes, se omite...")

            return

        def createTestLikes(N_RESTS):

            def topNfn(data, prev_set=None,numb_rests=None):
                '''Para cada LIKE de dev, generar 99 aleatorios no vistos en el conjunto previo'''

                new_items = 99

                id_r = data['id_restaurant'].values[0]
                id_u = data['id_user'].values[0]

                used_rests = [id_r]
                used_rests.extend(prev_set.loc[prev_set.id_user == id_u].id_restaurant.unique().tolist())

                # Lista de restaurantes no vistos
                available_rests = list(set(range(numb_rests)) - set(used_rests))
                assert len(available_rests) >=99
                new_rests = random.choices(available_rests, k=new_items)

                tmp = pd.DataFrame()
                tmp["id_restaurant"] = new_rests
                tmp["id_user"] = id_u
                tmp['is_dev'] = 0
                tmp['id_top'] = id_r #Para diferenciarlo en caso de que existan TOPS repetidos

                tmp = tmp.append({"id_restaurant":id_r, "id_user":id_u,"is_dev":1,"id_top":id_r,}, ignore_index=True)

                return tmp

            tqdm.pandas(desc="Creating like test sets")

            if(not os.path.exists(file_path+"DEV_LIKE")):
                DEV_L = self.getPickle(split_file_path_like, "DEV")
                DEV_T = self.getPickle(split_file_path_take, "DEV")
                DEV = DEV_L.append(DEV_T.loc[DEV_T.like==1],ignore_index=True)

                TRAIN = self.getPickle(file_path, "TRAIN_LIKE")

                DEV = DEV.groupby(["id_user", "id_restaurant"]).progress_apply(topNfn,prev_set=TRAIN, numb_rests=N_RESTS).reset_index(drop=True)
                self.toPickle(file_path, "DEV_LIKE", DEV); del DEV

            else: self.printW("Ya se ha creado el conjunto de DEV de likes, se omite...")

            if(not os.path.exists(file_path+"TEST_LIKE")):
                TEST_L = self.getPickle(split_file_path_like, "TEST")
                TEST_T = self.getPickle(split_file_path_take, "TEST")
                TEST = TEST_L.append(TEST_T.loc[TEST_T.like==1],ignore_index=True)

                TRAIN_DEV = self.getPickle(file_path, "TRAIN_DEV_LIKE")

                TEST = TEST.loc[TEST.like==1] #Solo los positivos para este caso
                TEST = TEST.groupby(["id_user", "id_restaurant"]).progress_apply(topNfn,prev_set=TRAIN_DEV, numb_rests=N_RESTS).reset_index(drop=True)
                self.toPickle(file_path, "TEST_LIKE", TEST);del TEST

            else: self.printW("Ya se ha creado el conjunto de TEST de likes, se omite...")

            return

        ################################################################################################################

        # Mirar si ya existen los datos
        # --------------------------------------------------------------------------------------------------------------

        file_path = self.PATH + self.MODEL_NAME.upper()
        split_file_path_take = file_path + "/original_take/"
        split_file_path_like = file_path + "/original_like/"

        file_path += "/data_" + str(self.CONFIG['neg_images']) + "_"+self.CONFIG["neg_likes"]+"/"

        RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();

        IMG["id_img"] = IMG.index

        URLS = RVW[["reviewId", "images"]].merge(IMG, left_on="reviewId", right_on="review")
        URLS["url"] = URLS.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)
        URLS = URLS[["id_img","url"]]

        RVW = RVW.drop(columns=['restaurantId', 'userId', 'url', 'text', 'title', "date" , 'language', 'rating', 'images'])

        #Para DEBUG (Ver url de fotos)---------------------------------------------------------------------------------
        #RVW = RVW.merge(IMG, left_on="reviewId", right_on="review")
        #RVW["url"] = RVW.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)

        if (os.path.exists(file_path) and len(os.listdir(file_path)) == 11):

            self.printW("Cargando datos generados previamente...")

            TRAIN_LIKE = self.getPickle(file_path, "TRAIN_LIKE")
            TRAIN_IMG = self.getPickle(file_path, "TRAIN_IMG")

            TRAIN_DEV_LIKE = self.getPickle(file_path, "TRAIN_DEV_LIKE")
            TRAIN_DEV_IMG = self.getPickle(file_path, "TRAIN_DEV_IMG")

            DEV_LIKE = self.getPickle(file_path, "DEV_LIKE")
            DEV_IMG = self.getPickle(file_path, "DEV_IMG")

            TEST_LIKE = self.getPickle(file_path, "TEST_LIKE")
            TEST_IMG = self.getPickle(file_path, "TEST_IMG")

            REST_TMP = self.getPickle(file_path, "REST_TMP")
            USR_TMP = self.getPickle(file_path, "USR_TMP")
            V_IMG = self.getPickle(file_path, "V_IMG")

            return {"TRAIN_LIKE": TRAIN_LIKE, "TRAIN_IMG": TRAIN_IMG, "TRAIN_DEV_LIKE": TRAIN_DEV_LIKE, "TRAIN_DEV_IMG": TRAIN_DEV_IMG,
                    "DEV_LIKE": DEV_LIKE, "DEV_IMG": DEV_IMG,"TEST_LIKE": TEST_LIKE, "TEST_IMG": TEST_IMG,
                    "IMG":IMG, "URLS":URLS, "N_RST":REST_TMP,"N_USR":USR_TMP, "V_IMG":V_IMG}

        os.makedirs(file_path, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------
        # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
        # --------------------------------------------------------------------------------------------------------------

        RVW = dropMultipleVisits(RVW)

        # Separar reviews con y sin imágen.
        # ---------------------------------------------------------------------------------------------------------------

        RVW_IM = RVW.loc[(RVW.num_images > 0)]  # Reviews sin imagen
        RVW_IM = RVW_IM.drop(columns=["num_images"])

        RVW_NI = RVW.loc[(RVW.num_images == 0)]  # Reviews con imagen
        RVW_NI = RVW_NI.drop(columns=["num_images"])

        # Mover ejemplos positivos a donde corresponde (N, 1, 1)
        # --------------------------------------------------------------------------------------------------------------

        splitIMGSets(split_file_path_take, RVW_IM)
        splitLIKESets(split_file_path_like, RVW_NI) #Como el anterior, pero solo mueve likes a DEV y TEST


        # Crear conjuntos de TRAIN y GUARDAR
        # --------------------------------------------------------------------------------------------------------------

        createTRAIN_IMG(IMG)
        createTRAIN_LIKE(len(REST_TMP),len(USR_TMP))

        # Crear los dos conjuntos de DEV y los dos de TEST (uno para las imágenes y otro para el TOP-N)
        # --------------------------------------------------------------------------------------------------------------

        createTestImages(IMG) #Crear los conjuntos para las imágenes
        createTestLikes(len(REST_TMP)) #Crear los conjuntos del TOP-N

        # ALMACENAR PICKLE ------------------------------------------------------------------------------------------------

        self.toPickle(file_path, "REST_TMP", len(REST_TMP))
        self.toPickle(file_path, "USR_TMP", len(USR_TMP))
        self.toPickle(file_path, "V_IMG", len(IMG.iloc[0].vector))

        # Cargar datos creados ------------------------------------------------------------------------------------------------

        TRAIN_LIKE = self.getPickle(file_path,"TRAIN_LIKE")
        TRAIN_IMG = self.getPickle(file_path,"TRAIN_IMG")

        TRAIN_DEV_LIKE = self.getPickle(file_path,"TRAIN_DEV_LIKE")
        TRAIN_DEV_IMG = self.getPickle(file_path,"TRAIN_DEV_IMG")

        DEV_LIKE = self.getPickle(file_path,"DEV_LIKE")
        DEV_IMG = self.getPickle(file_path,"DEV_IMG")

        TEST_LIKE = self.getPickle(file_path,"TEST_LIKE")
        TEST_IMG = self.getPickle(file_path,"TEST_IMG")

        return {"TRAIN_LIKE": TRAIN_LIKE, "TRAIN_IMG": TRAIN_IMG, "TRAIN_DEV_LIKE": TRAIN_DEV_LIKE, "TRAIN_DEV_IMG": TRAIN_DEV_IMG,
                "DEV_LIKE": DEV_LIKE, "DEV_IMG": DEV_IMG,"TEST_LIKE": TEST_LIKE, "TEST_IMG": TEST_IMG,
                "IMG": IMG, "URLS": URLS, "N_RST": len(REST_TMP),"N_USR": len(USR_TMP), "V_IMG": len(IMG.iloc[0].vector)}

    def getDataStats(self):

        def getNumbers(data, title=""):
            print(title.upper() + "\t" + "# USR\t" + str(len(data.id_user.unique())))
            print(title.upper() + "\t" + "# RST\t" + str(len(data.id_restaurant.unique())))
            print(title.upper() + "\t" + "# FTS\t" + str(len(data.id_img.unique())))

        file_path = self.PATH + self.MODEL_NAME.upper()[:-1]
        split_file_path_take = file_path + "/original_take/"
        split_file_path_like = file_path + "/original_like/"
        file_path += "/data_" + str(self.CONFIG['neg_images']) + "/"

        print("\n")
        print(self.CITY.upper())
        print("-"*50)

        TLK = self.DATA["TRAIN_LIKE"]
        TIM = self.DATA["TRAIN_IMG"]

        print("LIKE\tNO_LIKE")
        print(str(len(TLK.loc[TLK.like==1]))+"\t"+str(len(TLK.loc[TLK.like==0])))
        print("TAKE\tNO_TAKE")
        print(str(len(TIM.loc[TIM['take']== 1])) + "\t" + str(len(TIM.loc[TIM['take'] == 0])))

        print("-"*50)

