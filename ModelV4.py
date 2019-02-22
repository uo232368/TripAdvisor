# -*- coding: utf-8 -*-
from ModelClass import *

########################################################################################################################

class ModelV4(ModelClass):

    def __init__(self,city,option,config,seed = 2):

        modelName= "modelv4" #IMAGE-PREFERENCES
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

            user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            rest_input = tf.placeholder(tf.int32, shape=[None, 1], name="rest_input")
            img_input_best = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input_best")
            img_input_worst  = tf.placeholder(tf.float32, shape=[None, self.V_IMG], name="img_input_worst")

            # Embeddings -----------------------------------------------------------------------------------------------------------------

            E1 = tf.Variable(tf.truncated_normal([self.N_USR, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E1")
            E2 = tf.Variable(tf.truncated_normal([self.N_RST, emb_size], mean=0.0, stddev=1.0 / math.sqrt(emb_size)),name="E2")
            E3 = tf.Variable(tf.truncated_normal([self.V_IMG, emb_size*2], mean=0.0, stddev=1.0 / math.sqrt(emb_size*2)),name="E3")

            # Operaciones -----------------------------------------------------------------------------------------------------------------

            user_emb = tf.nn.embedding_lookup(E1, user_input[:,0])
            rest_emb = tf.nn.embedding_lookup(E2, rest_input[:,0])

            img_emb_best = tf.matmul(img_input_best, E3, name='img_emb_best')
            img_emb_worst = tf.matmul(img_input_worst, E3, name='img_emb_worst')

            user_rest_emb = tf.concat([user_emb,rest_emb],axis=1, name="user_rest_emb")

            # Cálculo de LOSS y optimizador ----------------------------------------------------------------------------------------------

            dot_best = tf.reduce_sum(tf.multiply(user_rest_emb, img_emb_best), 1, keepdims=True, name="dot_best")
            dot_worst = tf.reduce_sum(tf.multiply(user_rest_emb, img_emb_worst), 1, keepdims=True, name="dot_worst")

            batch_loss = tf.math.maximum(0.0,1-(dot_best-dot_worst), name="batch_loss")
            loss = tf.reduce_sum(batch_loss)

            adam = tf.train.AdamOptimizer(name='train_step_bin',learning_rate=self.CONFIG['learning_rate'])
            train_step_bin = adam.minimize(loss=loss, global_step=global_step_bin)

            # Crear objeto encargado de almacenar la red
            saver = tf.train.Saver(max_to_keep=1)

        return graph

    def train(self):

        train_df = pd.DataFrame()

        train_bin_loss = []
        train_bin_batches = np.array_split(self.TRAIN, len(self.TRAIN) // self.CONFIG['batch_size'])

        for bn in range(len(train_bin_batches)):

            batch_dtfm_imgs_ret = train_bin_batches[bn][['id_user', 'id_restaurant','vector','worst']]
            ret = batch_dtfm_imgs_ret.copy()
            batch_train_bin = batch_dtfm_imgs_ret.values

            feed_dict_bin = {
                "user_input:0": batch_train_bin[:, [0]],
                "rest_input:0": batch_train_bin[:, [1]],
                "img_input_best:0": np.row_stack(batch_train_bin[:, [2]][:, 0]),
                "img_input_worst:0": np.row_stack(batch_train_bin[:, [3]][:, 0])
            }

            _, batch_loss,dot_best, dot_worst = self.SESSION.run(['train_step_bin:0', 'batch_loss:0','dot_best:0','dot_worst:0'], feed_dict=feed_dict_bin)

            train_bin_loss.extend(batch_loss[:, 0])
            ret['best'] = dot_best[:, 0]
            ret['worst'] = dot_worst[:, 0]
            ret['loss'] = batch_loss[:, 0]

            train_df = train_df.append(ret)

        return (np.mean(train_bin_loss))

    def dev(self):

        def chooseCNTandRND():

            RST_CNTS = pd.DataFrame(columns=["id_restaurant", "vector"])
            RST_RNDM = pd.DataFrame(columns=["id_restaurant", "vector"])

            for i, g in self.TRAIN.groupby("id_restaurant"):
                all_c = np.row_stack(g.vector.values)

                cnt = np.mean(all_c, axis=0)
                dsts = scipy.spatial.distance.cdist([cnt], all_c, 'euclidean')
                indx = np.argmin(dsts)
                item_c = all_c[np.argmin(dsts), :]

                RST_CNTS = RST_CNTS.append({"id_restaurant": i, "vector": item_c}, ignore_index=True)
                RST_RNDM = RST_RNDM.append({"id_restaurant": i, "vector": g.sample(1)['vector'].values[0]},ignore_index=True)

            return RST_CNTS,RST_RNDM

        def getDist(dev,r,rnd=None, cnt=None,mode = "best"):

            if (len(dev) > 1): dev = np.row_stack(dev)
            else: dev = [dev[0]]

            if(mode == "best"):
                item = r.iloc[-1,:]['vector']

                '''

                #-------------------------------------------------------------------------------------------------------
                #if (self.CURRENT_EPOCH == 2 and r.id_user.values[0]==1290 and r.id_restaurant.values[0]==4084):
                    #print(r)

                all_c = r.vector.values
                if (len(all_c) > 1):
                    all_c = np.row_stack(all_c)
                else:
                    all_c = [all_c[0]]

                cnt = np.mean(all_c, axis=0)
                dsts = scipy.spatial.distance.cdist([cnt], all_c, 'euclidean')
                indx = np.argmin(dsts)
                item_c = all_c[np.argmin(dsts), :]

                dsts = scipy.spatial.distance.cdist(dev, [item], 'euclidean')
                min_dst_a = np.min(dsts)

                dsts = scipy.spatial.distance.cdist(dev, [item_c], 'euclidean')
                min_dst_b = np.min(dsts)

                if(min_dst_a>min_dst_b):
                    print("-"*30)
                    print("Review: ",r.reviewId.values[0])
                    print("Usuario: ",r.id_user.values[0])
                    print("Restaurante: ",r.id_restaurant.values[0])
                    print("")
                    print("Modelo: ", item)
                    print("Centroide: ", item_c)
                    print("Modelo: ", min_dst_a)
                    print("Centroide: ", min_dst_b)

                #-------------------------------------------------------------------------------------------------------
                '''

            if (mode == "random"):
                assert rnd is not None
                if(len(rnd.loc[rnd.id_restaurant.isin(r.id_restaurant.unique())])==0):
                    print("asdqasd")
                item = rnd.loc[rnd.id_restaurant.isin(r.id_restaurant.unique())]['vector'].values[0]

                #item = r.sample(1)['vector'].values[0]

            if (mode == "centroid"):
                assert cnt is not None
                if (len(cnt.loc[cnt.id_restaurant.isin(r.id_restaurant.unique())]) == 0):
                    print("asdqasd")
                item = cnt.loc[cnt.id_restaurant.isin(r.id_restaurant.unique())]['vector'].values[0]

                '''
                all = r.vector.values
                if (len(all) > 1): all = np.row_stack(all)
                else: all = [all[0]]

                cnt = np.mean(all, axis=0)
                dsts = scipy.spatial.distance.cdist([cnt], all, 'euclidean')
                indx = np.argmin(dsts)
                item = all[np.argmin(dsts), :]
                '''

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

            return np.min(dsts)

        # Recomendación de restaurantes --------------------------------------------------------------------------------

        dev_bin_res = pd.DataFrame()

        dev_loss = []

        dev_img_loss = []
        dev_img_loss_rndm = []
        dev_img_loss_cnt = []

        #dev_img_loss_max = []
        #dev_img_loss_min = []

        # Recomendación de imágenes --------------------------------------------------------------------------------
        dev_img_res = pd.DataFrame()

        for batch_di in np.array_split(self.DEV, 10):
            batch_dtfm_imgs_ret = batch_di.copy()
            batch_dtfm_imgs = batch_di[['id_user', 'id_restaurant','vector']]
            batch_dm = batch_dtfm_imgs.values

            feed_dict_bin = {
                "user_input:0": batch_dm[:, [0]],
                "rest_input:0": batch_dm[:, [1]],
                "img_input_best:0": np.row_stack(batch_dm[:, [2]][:, 0]),
            }

            #dot_best = self.SESSION.run('dot_best:0', feed_dict=feed_dict_bin)
            dot_best = self.SESSION.run('dot_best:0', feed_dict=feed_dict_bin)

            batch_dtfm_imgs_ret['prediction'] = dot_best[:, 0]
            dev_img_res = dev_img_res.append(batch_dtfm_imgs_ret, ignore_index=True)

        #Se obtinenen los centroides de los reataurantes y los random (CON LOS DATOS DE TRAIN Y SOLO LA PRIMERA EPOCH)
        if(self.CURRENT_EPOCH==0):RST_CNTS, RST_RNDM = chooseCNTandRND()

        for i,r in dev_img_res.groupby("reviewId"):
            r = r.sort_values("prediction")
            dev = r.loc[r.is_dev==1,'vector'].values

            '''
            RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();
            RST_IMGS = RVW.loc[RVW.id_restaurant == r.id_restaurant.values[0]]
            RST_IMGS = IMG.merge(RST_IMGS, left_on="review", right_on="reviewId")
            RST_IMGS_MTX = np.row_stack(RST_IMGS.vector.values)

            best_row = np.argmin(scipy.spatial.distance.cdist(RST_IMGS_MTX, [best], 'euclidean'))
            best_url = RST_IMGS.iloc[best_row,:].images[RST_IMGS.iloc[best_row,:].image-1]
            print(best_url['image_url_lowres'])

            dev_rows = np.argmin(scipy.spatial.distance.cdist(RST_IMGS_MTX, np.row_stack(dev), 'euclidean'), axis=0)
            #dev_items = RST_IMGS.iloc[dev_rows,:]
            dev_urls = RST_IMGS.iloc[dev_rows[0],:].images
            for u in dev_urls: print(u['image_url_lowres'])

            less = np.argmin(scipy.spatial.distance.cdist(np.row_stack(dev), [best], 'euclidean'))
            less_item = np.row_stack(dev)[less,:]
            dev_items = RST_IMGS.iloc[dev_rows,:]
            '''

            min_dst = getDist(dev,r,mode = "best")

            if(self.CURRENT_EPOCH==0):
                min_dst_rnd = getDist(dev,r,rnd = RST_RNDM, mode = "random")
                min_dst_cnt = getDist(dev,r,cnt = RST_CNTS, mode = "centroid")
                #min_dst_max, _ = getDist(dev,r,mode = "max")
                #min_dst_min, _ = getDist(dev,r,mode = "min")

            else: min_dst_rnd = min_dst_cnt = min_dst_max = min_dst_min = -1

            dev_img_loss.append(min_dst)
            dev_img_loss_rndm.append(min_dst_rnd)
            dev_img_loss_cnt.append(min_dst_cnt)
            #dev_img_loss_max.append(min_dst_max)
            #dev_img_loss_min.append(min_dst_min)


        return ((dev_img_loss,dev_img_loss_rndm,dev_img_loss_cnt,dev_img_loss_max,dev_img_loss_min), min_dst)

    def gridSearchPrint(self,epoch,train,dev):

        if(epoch==0):
            header = ["E","T_LOSS","MIN_D","RNDM","CNTR","MAX  ","MIN  ","MIN_POS"]
            header_line = "\t".join(header)
            print(header_line)

        log_items = [epoch+1]
        log_items.extend(np.round([train], decimals=4))
        log_items.extend(np.round([np.mean(dev[0]),np.mean(dev[1]),np.mean(dev[2]),np.mean(dev[3]),np.mean(dev[4]),np.sum(dev[5])],decimals=4))

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


        # Eliminar usuarios con menos de min_usr_rvws
        # ---------------------------------------------------------------------------------------------------------------

        USR_LST = RVW.groupby("userId", as_index=False).count()
        USR_LST = USR_LST.loc[(USR_LST.like >= self.CONFIG['min_usr_rvws']), "userId"].values
        RVW = RVW.loc[RVW.userId.isin(USR_LST)]


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
            print("USUARIOS:"+str(len(RVW.userId.unique())))
            print("RESTAURANTES:"+str(len(RVW.restaurantId.unique())))


        return RVW, IMG, USR_TMP,REST_TMP

    def getData(self):

        # Mirar si ya existen los datos
        # ---------------------------------------------------------------------------------------------------------------

        file_path = self.PATH + self.MODEL_NAME.upper() + "/data"
        file_path += "_" + str(self.CONFIG['min_usr_rvws'])
        file_path += "_" + str(self.CONFIG['neg_examples'])
        file_path += "/"


        if (os.path.exists(file_path)):
            self.printW("Cargando datos generados previamente...")

            TRAIN_v4 = self.getPickle(file_path, "TRAIN_v4")
            DEV_v4 = self.getPickle(file_path, "DEV_v4")
            TEST_v4 = self.getPickle(file_path, "TEST_v4")

            REST_TMP = self.getPickle(file_path, "REST_TMP")
            USR_TMP = self.getPickle(file_path, "USR_TMP")
            V_IMG = self.getPickle(file_path, "V_IMG")

            return (TRAIN_v4, DEV_v4, TEST_v4, REST_TMP, USR_TMP, V_IMG)



        # ---------------------------------------------------------------------------------------------------------------

        RVW, IMG, USR_TMP, REST_TMP = self.getFilteredData();
        RVW = RVW.drop(columns=['restaurantId', 'userId', 'url', 'text', 'title', 'date', 'images', 'rating', 'language', 'like'])

        # Mover ejemplos positivos a donde corresponde (N, 1, 1)
        # ---------------------------------------------------------------------------------------------------------------

        TRAIN = pd.DataFrame()
        DEV = pd.DataFrame()
        TEST = pd.DataFrame()

        def split_fn(d):

            items = len(d)

            dev_test_items = 1
            train_items = items - (2 * dev_test_items)

            d["TO_TRAIN"] = 0;
            d["TO_DEV"] = 0;
            d["TO_TEST"] = 0

            d.iloc[:train_items, -3] = 1
            d.iloc[train_items:train_items + dev_test_items, -2] = 1
            d.iloc[train_items + dev_test_items:, -1] = 1

            return d

        RVW_S = RVW.groupby('id_user').apply(split_fn)

        TRAIN_v4 = RVW_S.loc[RVW_S.TO_TRAIN == 1]
        DEV_v4 = RVW_S.loc[RVW_S.TO_DEV == 1]
        TEST_v4 = RVW_S.loc[RVW_S.TO_TEST == 1]

        TRAIN_v4 = TRAIN_v4.drop(columns=["TO_TRAIN", "TO_DEV", "TO_TEST"])
        DEV_v4 = DEV_v4.drop(columns=["TO_TRAIN", "TO_DEV", "TO_TEST"])
        TEST_v4 = TEST_v4.drop(columns=["TO_TRAIN", "TO_DEV", "TO_TEST"])

        # Añadir los vectores de la imágen a los ejemplos
        # ---------------------------------------------------------------------------------------------------------------
        TRAIN_v4 = IMG.merge(TRAIN_v4, left_on='review', right_on='reviewId', how='inner')
        DEV_v4 = IMG.merge(DEV_v4, left_on='review', right_on='reviewId', how='inner')
        TEST_v4 = IMG.merge(TEST_v4, left_on='review', right_on='reviewId', how='inner')

        TRAIN_v4 = TRAIN_v4.drop(columns=["image", "num_images"])
        DEV_v4 = DEV_v4.drop(columns=["image", "num_images"])
        TEST_v4 = TEST_v4.drop(columns=["image", "num_images"])

        # Añadir ejemplos negativos en TRAIN_V4 DEV_V4 y TEST_V4
        # --------------------------------------------------------------------------------------------------------------

        IMGS = IMG.merge(RVW[["id_restaurant", "id_user", "reviewId"]], left_on='review', right_on='reviewId', how='inner')

        #RVW_2, IMG_2, _, _ = self.getFilteredData();

        def trainFn(data):

            neg = self.CONFIG['neg_examples']

            id_r = data.id_restaurant.values[0]
            id_u = data.id_user.values[0]
            id_rv = data.reviewId.values[0]

            img_best = data.vector.values[0]

            images = IMGS.loc[(IMGS.id_restaurant == id_r) & (IMGS.id_user != id_u), "vector"]

            '''
            if(id_rv==557715079):
                dir = "tmp_img/" + str(id_r)
                os.makedirs(dir, exist_ok=True)

                im_tst = IMGS.loc[(IMGS.id_restaurant == id_r)]
                im_tst["real"] = 0
                im_tst.loc[im_tst.id_user==id_u, "real"] = 1
                im_tmp = RVW_2[['images',"reviewId"]].merge(im_tst, left_on='reviewId', right_on='reviewId')
                im_tmp["images"] = im_tmp.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)

                dists = scipy.spatial.distance.cdist([img_best], np.row_stack(im_tmp.vector.values), "euclidean")
                im_tmp["dists"] = dists[0]
                im_tmp = im_tmp.sort_values("dists")
                im_tmp.iloc[-neg:, im_tmp.columns.get_loc("real")] = 2

                im_tmp["name"]= im_tmp.images.apply(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest())

                print(im_tmp[["name", "dists", "vector"]])

            '''
            '''
            if(len(images)>15 and False):

                dir = "tmp_img/"+str(id_r)
                os.makedirs(dir,exist_ok=True )

                im_tst = IMGS.loc[(IMGS.id_restaurant == id_r)]
                im_tst["real"] = 0
                im_tst.loc[im_tst.id_user==id_u, "real"] = 1
                im_tmp = RVW_2[['images',"reviewId"]].merge(im_tst, left_on='reviewId', right_on='reviewId')
                im_tmp["images"] = im_tmp.apply(lambda x: x.images[x.image - 1]['image_url_lowres'], axis=1)

                dists = scipy.spatial.distance.cdist([img_best], np.row_stack(im_tmp.vector.values), "euclidean")
                im_tmp["dists"] = dists[0]
                im_tmp = im_tmp.sort_values("dists")
                im_tmp.iloc[-neg:, im_tmp.columns.get_loc("real")] = 2

                im_tmp["name"]= im_tmp.images.apply(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest())

                S = im_tmp[["name", "dists"]].to_csv("dists-"+str(id_r)+".csv")
                D = scipy.spatial.distance.cdist(np.row_stack(im_tmp.vector.values), np.row_stack(im_tmp.vector.values),"euclidean").to_csv("all-dists-"+str(id_r)+".csv")
                D = pd.DataFrame(scipy.spatial.distance.cdist(np.row_stack(im_tmp.vector.values), np.row_stack(im_tmp.vector.values),"euclidean"))
                D.to_csv("all-dists-" + str(id_r) + ".csv")

                for i,d in im_tmp.iterrows():
                    name = hashlib.md5(str(d.images).encode('utf-8')).hexdigest()
                    if(d['real']==1):
                        if(d['dists']==0):name="r-"+name;
                        else:name="d-"+name;
                    elif (d['real'] == 2):
                        name="lj-"+name;

                    urllib.request.urlretrieve(d.images, dir+"/"+name+".jpg")

            '''

            if (len(images) == 0): return

            images = np.row_stack(images.values)

            dists = scipy.spatial.distance.cdist([img_best], images, "euclidean")
            indx = np.argsort(dists)[0][-neg:]
            img_worst = images[indx, :]

            ret = pd.DataFrame([data.squeeze()] * len(img_worst))
            ret["worst"] = img_worst.tolist()
            ret = ret.drop(columns=["item"])

            return (ret)

        #Al train se añaden ejemplos n del tipo (u,r, foto de u en r, foto de otro u en r) [las n fotos más lejanas de la del usuario]
        TRAIN_v4["item"] = TRAIN_v4.index
        TRAIN_v4 = TRAIN_v4.groupby("item").apply(trainFn).reset_index(drop=True)

        def testFn(data):
            id_r = data.id_restaurant.values[0]
            id_u = data.id_user.values[0]
            id_rev = data.reviewId.values[0]

            tmp = IMGS.loc[IMGS.id_restaurant == id_r]
            #images = IMGS.loc[(IMGS.id_restaurant == id_r) & (IMGS.id_user != id_u), "vector"]

            tmp['is_dev'] = 0
            tmp['like'] = 0
            tmp['id_user'] = id_u

            tmp.loc[tmp.reviewId.isin(data.reviewId.values), "is_dev"] = 1

            tmp['reviewId'] = id_rev

            tmp = tmp[['reviewId','id_user','id_restaurant','vector','is_dev']]

            return tmp

        #ToDo: SOLO RESTAURANTES EN TRAIN QUE TENGAN 5 FOTOS?

        # Obtener los restaurantes con 5 o más imágenes en total.
        RST_DEV = RVW.groupby("id_restaurant").apply(lambda x: pd.Series({"id_restaurant": x.id_restaurant.values[0], "imgs": sum(x.num_images.values)}))
        RST_DEV = RST_DEV.loc[RST_DEV.imgs >= 5, "id_restaurant"].values

        #Utilizando solo los restaurantes con más de 5 fotos en total, al DEV y TEST se añaden, para cada ejemplo, el resto de fotos del restaurante.
        DEV_v4 = DEV_v4.loc[DEV_v4.id_restaurant.isin(RST_DEV)]
        DEV_v4 = DEV_v4.groupby("reviewId").apply(testFn).reset_index(drop=True)

        TEST_v4 = TEST_v4.loc[TEST_v4.id_restaurant.isin(RST_DEV)]
        TEST_v4 = TEST_v4.groupby("reviewId").apply(testFn).reset_index(drop=True)

        # MEZCLAR DATOS ------------------------------------------------------------------------------------------------

        TRAIN_v4 = utils.shuffle(TRAIN_v4, random_state=self.SEED).reset_index(drop=True)

        # ALMACENAR PICKLE ------------------------------------------------------------------------------------------------

        os.makedirs(file_path)

        self.toPickle(file_path, "TRAIN_v4", TRAIN_v4)
        self.toPickle(file_path, "DEV_v4", DEV_v4)
        self.toPickle(file_path, "TEST_v4", TEST_v4)

        self.toPickle(file_path, "REST_TMP", len(REST_TMP))
        self.toPickle(file_path, "USR_TMP", len(USR_TMP))
        self.toPickle(file_path, "V_IMG", len(IMG.iloc[0].vector))

        return (TRAIN_v4, DEV_v4, TEST_v4, len(REST_TMP), len(USR_TMP), len(IMG.iloc[0].vector))
