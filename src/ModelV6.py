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
            #config = tf.ConfigProto()
            #config.gpu_options.allow_growth = True
            #sess = tf.Session(config=config)
            #keras.backend.set_session(sess)

            cfg = K.tf.ConfigProto()
            cfg.gpu_options.allow_growth = True
            sess = K.tf.Session(config=cfg)
            K.set_session(sess)

            self.SESSION = sess

            #Conjuntos de entrenamiento
            keras.backend.get_session().run(tf.global_variables_initializer())

            train_sequence_take = self.TrainSequenceTake(self.DATA["TRAIN_IMG"],self.CONFIG['batch_size'], self)
            train_sequence_like = self.TrainSequenceLike(self.DATA["TRAIN_LIKE"],self.CONFIG['batch_size'], self)

            dev_sequence_take   = self.DevSequenceTake(self.DATA["DEV_IMG"], self.CONFIG['batch_size'], self)
            dev_sequence_like   = self.DevSequenceLike(self.DATA["DEV_LIKE"], self.CONFIG['batch_size'], self)

            #self.printE("CAMBIAR ESTO")
            #dev_sequence_take   = self.DevSequenceTake(self.DATA["TEST_IMG"], self.CONFIG['batch_size'], self)
            #dev_sequence_like   = self.DevSequenceLike(self.DATA["TEST_LIKE"], self.CONFIG['batch_size'], self)

            for e in range(max_epochs):
                tes = time.time()

                self.CURRENT_EPOCH = e

                train_ret = self.train(train_sequence_take, train_sequence_like)

                if(e!=max_epochs-1):self.gridSearchPrint(e,time.time()-tes, train_ret, {})

            dev_ret = self.dev(dev_sequence_take,dev_sequence_like)
            self.gridSearchPrint(e,time.time()-tes, train_ret, dev_ret)

            K.clear_session()

    def getImagePos(self, data):

        def getPos(r):
            r = r.reset_index(drop=True)
            id_r = r.id_restaurant.unique()[0]
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            return pos

        pcnt1_model = []
        ret = []

        for i, r in data.groupby("id_test"):
            r = r.sort_values("prediction", ascending=False)
            item_pos = getPos(r)

            pcnt1_model.append((item_pos - 1) / len(r))

            usr = r.id_user.values[0]
            rst = r.id_restaurant.values[0]

            ret.append([usr,rst,i,item_pos,len(r),len(r.loc[r.is_dev == 1]),pcnt1_model[-1]])

        ret = pd.DataFrame(ret)
        ret.columns=["id_user", "id_restaurant","id_test","model_pos","n_photos","n_photos_dev","PCNT-1_MDL"]

        return ret,np.mean(pcnt1_model),np.median(pcnt1_model)

    def getTopN(self, data):

        def getPos(r):
            r = r.reset_index(drop=True)
            #pos = len(r)-max(r.loc[r.is_dev == 1].index.values)
            pos = min(r.loc[r.is_dev == 1].index.values)+1

            if((r.id_user.values[0]==289 and r.id_top.values[0]==110) or
                (r.id_user.values[0]==205 and r.id_top.values[0]==437)):
                print(r.loc[r.is_dev == 1].prediction.values[0])

            return pos

        ret = []

        grp = data.groupby(["id_user","id_top"])

        for i, r in grp:
            r = r.sort_values("prediction", ascending=False)

            item_pos = getPos(r)

            ret.append([i[0],i[1],item_pos])

        ret = pd.DataFrame(ret)
        ret.columns=["id_user", "id_restaurant","model_pos"]

        #for t in self.CONFIG["top"]:
        #    ret[str(t)] = (ret["model_pos"]<=t).astype(int)

        #ret2 = ret.iloc[:, 1:].sum(axis=0).reset_index()
        #print(((ret2[0] / len(grp)) * 100).values)

        return ret,np.mean(ret.model_pos), np.median(ret.model_pos)

    def gridSearchPrint(self,epoch,time,train,dev):

        def frm(item):
            return'{:6.3f}'.format(item)

        #--------------------------------------------------------------------------------------------------------------
        header = ["E", "E_TIME"]
        header.extend(train.keys())
        header.extend(dev.keys())

        if(epoch==0):print("\t".join(header))

        line = [time]
        line.extend(train.values())
        line.extend(dev.values())
        line = list(map(frm,line))
        line.insert(0,str(epoch))

        print("\t".join(line))

        return None

    def finalTrain(self, epochs = 1, save=False):

        if(os.path.exists(self.MODEL_PATH)):
            self.printE("Ya existe el modelo...")
            exit()

        # Conjuntos de entrenamiento

        train_dev_sequence_take = self.TrainSequenceTake(self.DATA["TRAIN_DEV_IMG"], self.CONFIG['batch_size'], self)
        train_dev_sequence_like = self.TrainSequenceLike(self.DATA["TRAIN_DEV_LIKE"], self.CONFIG['batch_size'], self)

        #train_dev_sequence_take = self.TrainSequenceTake(self.DATA["TRAIN_IMG"], self.CONFIG['batch_size'], self)
        #train_dev_sequence_like = self.TrainSequenceLike(self.DATA["TRAIN_LIKE"], self.CONFIG['batch_size'], self)

        test_sequence_take = self.DevSequenceTake(self.DATA["TEST_IMG"], self.CONFIG['batch_size'], self)
        test_sequence_like = self.DevSequenceLike(self.DATA["TEST_LIKE"], self.CONFIG['batch_size'], self)

        '''
        usrs = [289, 205];
        rsts = [110, 437]
        items = self.DATA["TEST_LIKE"].loc[(self.DATA["TEST_LIKE"].id_user.isin(usrs)) & (self.DATA["TEST_LIKE"].id_top.isin(rsts))]
        DEV = self.DATA["DEV_LIKE"]
        DEV = DEV.append(items, ignore_index=True)
        

        test_sequence_take = self.DevSequenceTake(self.DATA["DEV_IMG"], self.CONFIG['batch_size'], self)
        test_sequence_like = self.DevSequenceLike(DEV, self.CONFIG['batch_size'], self)
        '''

        # Crear el modelo
        self.MODEL = self.getModel()

        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        sess = K.tf.Session(config=cfg)
        K.set_session(sess)

        self.SESSION = sess

        keras.backend.get_session().run(tf.global_variables_initializer())

        for e in range(epochs):
            print(e,"/",epochs)

            tes = time.time()
            self.CURRENT_EPOCH = e

            train_ret = self.train(train_dev_sequence_take, train_dev_sequence_like)

        dev_ret = self.dev(test_sequence_take, test_sequence_like)
        self.gridSearchPrint(e, time.time() - tes, train_ret, dev_ret)

        if(save):
            os.makedirs(self.MODEL_PATH)
            for m in self.MODEL:
                m.save(self.MODEL_PATH+"/"+m.name)

        K.clear_session()

    def takeBaseline(self, test=False):

        def getTrain():

            file_path = "/media/HDD/pperez/TripAdvisor/" + self.CITY.lower() + "_data/MODELV6/"
            path = file_path + "original_take/"

            if (test == False):
                TRAIN = self.getPickle(path, "TRAIN")
            else:
                TRAIN = self.getPickle(path, "TRAIN_DEV")

            # Añadir las imágenes a TRAIN
            _, IMG, _, _ = self.getFilteredData();
            IMG["id_img"] = IMG.index
            TRAIN = TRAIN.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
            TRAIN = TRAIN.drop(columns=['review'])

            return TRAIN

        def getCentroids(TRAIN):

            RST_CNTS = pd.DataFrame(columns=["id_restaurant", "vector"])

            for i, g in TRAIN.groupby("id_restaurant"):
                all_c = self.DATA["IMG"][g.id_img.values, :]
                cnt = np.mean(all_c, axis=0)

                RST_CNTS = RST_CNTS.append({"id_restaurant": i, "vector": cnt}, ignore_index=True)

            return RST_CNTS

        def getPos(data, ITEMS):
            g = data.copy()
            id_rest = g.id_restaurant.unique()[0]
            item = ITEMS.loc[ITEMS.id_restaurant == id_rest].vector.values[0]

            rst_imgs = self.DATA["IMG"][g.id_img.values, :]
            dsts = scipy.spatial.distance.cdist(rst_imgs, [item], 'euclidean')

            g["dsts"] = dsts

            g = g.sort_values("dsts").reset_index(drop=True)

            return min(g.loc[g.is_dev == 1].index.values) + 1

        def getRndPos(data):

            g = data.copy()
            pos = []

            g["prob"] = np.random.random_sample(len(g))
            g = g.sort_values("prob").reset_index(drop=True)

            return len(g) - max(g.loc[g.is_dev == 1].index.values)

        # Fijar semilla
        # ---------------------------------------------------------------------------------------------------------------
        np.random.seed(self.SEED)

        # Obtener el conjunto de TRAIN original
        # ---------------------------------------------------------------------------------------------------------------

        TRAIN = getTrain()

        # Obtener los centroides de los restaurantes en train y los random
        # ---------------------------------------------------------------------------------------------------------------

        RST_CNTS = getCentroids(TRAIN)

        # Para cada caso de DEV, calcular el resultado de los baselines
        # ---------------------------------------------------------------------------------------------------------------

        ret = []
        rpt = 10

        if (test == False):
            ITEMS = self.DATA["DEV_IMG"]
        else:
            ITEMS = self.DATA["TEST_IMG"]

        for i, g in ITEMS.groupby("id_test"):

            cnt_pos = getPos(g, RST_CNTS)
            rnd_pos = []

            for r in range(rpt): rnd_pos.append(getRndPos(g))

            d = [g.id_user.values[0],g.id_restaurant.values[0],i,len(g),cnt_pos]
            d.extend(rnd_pos)

            ret.append(d)

        RET = pd.DataFrame(ret)

        title = ["id_user", "id_restaurant","id_test","n_photos", "cnt_pos"]
        title.extend(list(map(lambda x: "rnd_pos_" + str(x), range(10))))

        RET.columns = title

        RET["PCNT-1_CNT"] = RET.apply(lambda x: (x.cnt_pos - 1) / x.n_photos, axis=1)
        for r in range(rpt): RET["PCNT-1_RND_"+str(r)] = RET.apply(lambda x: (x["rnd_pos_"+str(r)] - 1) / x.n_photos, axis=1)

        ret_path = "docs/" + self.DATE + "/Baselines"
        os.makedirs(ret_path, exist_ok=True)

        RET.to_excel(ret_path+"/Take_" + self.CITY + ("_TEST" if test else "_DEV") + ".xlsx")

        #print("RND\t%f\t%f" % (RET["PCNT-1_RND"].mean(), RET["PCNT-1_RND"].median()))
        avg = [];mdn = []
        for r in range(rpt): avg.append(RET["PCNT-1_RND_"+str(r)].mean()); mdn.append(RET["PCNT-1_RND_"+str(r)].median())

        print("RND\t%f\t%f" % (np.mean(avg), np.mean(mdn)))
        print("CNT\t%f\t%f" % (RET["PCNT-1_CNT"].mean(), RET["PCNT-1_CNT"].median()))

        return RET

    def likeBaseline(self, test=False):

        if (test == False):
            TRAIN = self.DATA["TRAIN_LIKE"]
            TRAIN = TRAIN.loc[TRAIN.reviewId>-1]

            TEST  = self.DATA["DEV_LIKE"]
        else:
            TRAIN = self.DATA["TRAIN_DEV_LIKE"]
            TRAIN = TRAIN.loc[TRAIN.reviewId>-1]

            TEST  = self.DATA["TEST_LIKE"]

        # Calcular la probabilidad de cada restaurante
        #REST = TRAIN.groupby("id_restaurant").apply(lambda x : pd.Series({"prediction":sum(x.like)/len(x)})).reset_index()
        REST = TRAIN.groupby("id_restaurant").apply(lambda x : pd.Series({"prediction":sum(x.like)})).reset_index()

        # Añadir probabilidades al TEST
        TEST = TEST.merge(REST, on="id_restaurant")

        # Calcular posiciones
        RET = []

        for i, r in TEST.groupby(["id_user","id_top"]):
            r = r.sort_values("prediction", ascending=False).reset_index()
            item_pos = r.loc[r.is_dev == 1].index.values[0]+1

            RET.append([i[0],i[1],item_pos])


        RET = pd.DataFrame(RET)
        RET.columns = ["id_user","id_top","baseline_pos"]


        # Obtener solo los usuarios y restaurantes involucrados
        TRAIN_USR = TRAIN.loc[TRAIN.id_user.isin(RET.id_user.unique())]
        TRAIN_RST = TRAIN.loc[TRAIN.id_restaurant.isin(RET.id_top.unique())]

        # Obtener datos de cada usuario y cada restaurante en TRAIN
        TRAIN_USR = TRAIN_USR.groupby("id_user").apply(lambda x: pd.Series({"usr_train_likes": x.like.sum(), "usr_train_reviews": len(x), "usr_train_nRest": len(x.id_restaurant.unique())})).reset_index()
        TRAIN_USR["usr_train_dislikes"] = TRAIN_USR.usr_train_reviews - TRAIN_USR.usr_train_likes

        TRAIN_RST = TRAIN_RST.groupby("id_restaurant").apply(lambda x: pd.Series({"rst_train_likes": x.like.sum(), "rst_train_reviews": len(x),"rst_train_nUsrs": len(x.id_user.unique())})).reset_index()
        TRAIN_RST["rst_train_dislikes"] = TRAIN_RST.rst_train_reviews - TRAIN_RST.rst_train_likes

        RET = RET.merge(TRAIN_USR, on="id_user")
        RET = RET.merge(TRAIN_RST, left_on="id_top", right_on="id_restaurant")

        RET = RET[["id_user", "id_restaurant", "baseline_pos",
                   "usr_train_likes", "usr_train_dislikes", "usr_train_reviews", "usr_train_nRest",
                   "rst_train_likes", "rst_train_dislikes", "rst_train_reviews", "rst_train_nUsrs"]]

        print(RET.baseline_pos.mean(),RET.baseline_pos.median())

        ret_path = "docs/" + self.DATE + "/Baselines"
        os.makedirs(ret_path, exist_ok=True)
        RET.to_excel(ret_path+"/Like_" + self.CITY + ("_TEST" if test else "_DEV") + ".xlsx")

        return RET

    def getDetailedResults(self):

        def getTrain():

            TRAIN = self.getPickle(self.DATA_PATH + "original_take/", "TRAIN_DEV")

            # Añadir las imágenes a TRAIN
            _, IMG, _, _ = self.getFilteredData();
            IMG["id_img"] = IMG.index
            TRAIN = TRAIN.merge(IMG[["review", "id_img"]], left_on="reviewId", right_on="review")
            TRAIN = TRAIN.drop(columns=['review'])

            TRAIN = TRAIN.groupby("id_user").apply(lambda x: pd.Series({"n_photos_train": len(x.id_img.unique()), "n_rest_train": len(x.id_restaurant.unique())})).reset_index()

            return TRAIN

        def getIMGProbs(model):

            test_sequence_take = self.DevSequenceTake(self.DATA["TEST_IMG"], self.CONFIG['batch_size'], self)

            dts = self.DATA["TEST_IMG"].copy()

            prs = model.predict_generator(test_sequence_take, use_multiprocessing=True, workers=12)

            dts["prediction"] = prs

            return dts

        def getLKEProbs(model):

            test_sequence_like = self.DevSequenceLike(self.DATA["TEST_LIKE"], self.CONFIG['batch_size'], self)

            dts = self.DATA["TEST_LIKE"].copy()

            prs = model.predict_generator(test_sequence_like, use_multiprocessing=True, workers=12)

            dts["prediction"] = prs

            return dts

        def getAcAverage(data):
            ret = []
            for i, v in enumerate(data):
                ret.append(np.mean(data[:i+1]))
            return ret

        def op(data, type="average"):
            if("average" in type): return str(np.mean(data.values))
            else: return str(np.median(data.values))

        #---------------------------------------------------------------------------------------------------------------

        # Cargar los modelos
        #---------------------------------------------------------------------------------------------------------------

        if(not os.path.exists(self.MODEL_PATH)): self.printE("No existe el modelo."); exit()
        model_like = load_model(self.MODEL_PATH+"/model_1")
        model_take = load_model(self.MODEL_PATH+"/model_2")


        #---------------------------------------------------------------------------------------------------------------
        # BORRAR -------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        '''
        TR = self.DATA["TRAIN_LIKE"]
        TD = self.DATA["TRAIN_DEV_LIKE"]
        DV = self.DATA["DEV_LIKE"]
        TS = self.DATA["TEST_LIKE"]

        file_path = self.PATH + self.MODEL_NAME.upper()[:-1]
        split_file_path_like = file_path + "/original_like_["+str(self.CONFIG['min_revs_usr'])+"|"+str(self.CONFIG['min_revs_rst'])+"]/"

        TR_O = self.getPickle(split_file_path_like,"TRAIN")
        TD_O = self.getPickle(split_file_path_like,"TRAIN_DEV")
        DV_O = self.getPickle(split_file_path_like,"DEV")
        TS_O = self.getPickle(split_file_path_like,"TEST")

        T_RST = TR.groupby("id_restaurant").apply(lambda x : pd.Series({"like":sum(x.like), "dislike":len(x)-sum(x.like)})).reset_index()
        TD_RST = TD.groupby("id_restaurant").apply(lambda x : pd.Series({"like":sum(x.like), "dislike":len(x)-sum(x.like)})).reset_index()

        T_USR = TR.groupby("id_user").apply(lambda x: pd.Series({"like": sum(x.like), "dislike": len(x) - sum(x.like)})).reset_index()
        TD_USR = TD.groupby("id_user").apply(lambda x: pd.Series({"like": sum(x.like), "dislike": len(x) - sum(x.like)})).reset_index()

        DV_R_STATS = DV.merge(T_RST, on="id_restaurant")
        TS_R_STATS = TS.merge(TD_RST, on="id_restaurant")

        DV_U_STATS = DV.merge(T_USR, on="id_user")
        TS_U_STATS = TS.merge(TD_USR, on="id_user")

        exit()
        '''
        #---------------------------------------------------------------------------------------------------------------
        # Evaluar en TEST
        #---------------------------------------------------------------------------------------------------------------

        RET = pd.DataFrame(columns=["id_user","id_restaurant","n_photos","n_photos_dev","model_pos","pcnt_model","pcnt1_model"])

        if(self.CONFIG["use_images"]==1):
            probs = getIMGProbs(model_take)
            pcnt,_,_ = self.getImagePos(probs)

            blines = self.takeBaseline(test=True)#[["id_user","id_test", "rnd_pos","cnt_pos","PCNT-1_CNT", "PCNT-1_RND"]]

            #Cargar todos los datos con los que se entrena (TRAIN+DEV)
            TRAIN_DATA = getTrain()

            RET = pcnt.merge(TRAIN_DATA, right_on="id_user", left_on="id_user")
            RET = RET.merge(blines, right_on="id_test", left_on="id_test")

            #RET = RET[['id_user_x', 'id_restaurant','n_rest_train', 'n_photos_train', 'n_photos', 'n_photos_dev', 'model_pos',"rnd_pos","cnt_pos", 'PCNT-1_RND', 'PCNT-1_CNT','PCNT-1_MDL']]

            #RET["RND-MOD"] = RET["PCNT-1_RND"] - RET["PCNT-1_MDL"]
            #RET["CNT-MOD"] = RET["PCNT-1_CNT"] - RET["PCNT-1_MDL"]

            RET = RET.sort_values("n_photos_train", ascending=False).reset_index(drop=True)

            #-----------------------------------------------------------------------------------------------------------
            # Eliminar los restaurantes que tengan menos de 10 imágenes en el test (son faciles de acertar)
            #-----------------------------------------------------------------------------------------------------------

            nRestIMG = 10

            RET10 = RET.loc[(RET["n_photos_x"] >= nRestIMG)]

            method = "median"

            random_cols = [col for col in RET10.columns if 'PCNT-1_RND_' in col]

            print("-" * 100)
            print(method.upper())
            print("-" * 100)
            print("N_FOTOS_TRAIN_USR (>=)\tN_ITEMS\t%ITEMS\tRANDOM\tCENTROIDE\tMODELO")
            for it in range(100,0,-1):
                data = RET10.loc[RET10["n_photos_train"] >= it]

                print(str(it) + "\t" +
                      str(len(data)) + "\t" +
                      str(len(data) / len(RET10)) + "\t" +
                      op(data[random_cols].mean(axis=1),type=method) + "\t" +
                      op(data["PCNT-1_CNT"],type=method) + "\t" +
                      op(data["PCNT-1_MDL"],type=method))

            print("-" * 100)

            #-----------------------------------------------------------------------------------------------------------
            # Eliminar los usuarios que tengan menos de 10 fotos en train (se sabe poco de ellos)
            #-----------------------------------------------------------------------------------------------------------

            nUserIMG = 10

            RET_f = RET10.loc[(RET10["n_photos_train"] >= nUserIMG)]

            #_,WCX_res = scipy.stats.wilcoxon(RET_f.model_pos, RET_f.rnd_pos)

            #print("Sign test p-values: " +str( WCX_res))

            #print("-" * 100)

            print("\tRND\tCNT\tMOD")

            positions = [1, 2, 3]

            random_cols = [col for col in RET10.columns if 'rnd_pos_' in col]

            for p in positions:
                rnd = np.average(np.sum(RET_f[random_cols] == p))
                #rnd = len(RET_f.loc[RET_f.rnd_pos==p])
                cnt = len(RET_f.loc[RET_f.cnt_pos==p])
                mdl = len(RET_f.loc[RET_f.model_pos==p])
                print("%d\t%f\t%d\t%d" % (p, rnd, cnt, mdl) )

            print("-" * 100)
            #print("RND\tCNT\tMOD")
            #print("%f\t%f\t%f" % (np.median(RET_f["PCNT-1_RND"]),np.median(RET_f["PCNT-1_CNT"]),np.median(RET_f["PCNT-1_MDL"])))

            RET.to_excel("docs/"+self.DATE+"/"+self.MODEL_NAME+"_"+self.CITY+"_TAKE.xls") # Por usuario

        if(self.CONFIG["use_like"]==1):

            probs = getLKEProbs(model_like)
            poss,_, _ = self.getTopN(probs)

            baseline = self.likeBaseline(test=True)

            RET = poss.merge(baseline, on=["id_user","id_restaurant"])

            #Cuantas reviews tiene cada restaurante ??
            positions = [1, 2, 3]

            for p in positions:
                data = RET.loc[RET.model_pos==p]
                data_b = RET.loc[RET.baseline_pos==p]
                print(p,"\t",len(data),"\t", len(data)/len(RET),"\t",len(data_b),"\t", len(data_b)/len(RET),"\t", len(RET))

            print("-" * 100)

            n_items = [100,90,80,70,60,50,40,30,20,10,5,4,3,2,1]

            #Para el número de reviews del restaurante
            for t in n_items:
                data = RET.loc[RET.rst_train_reviews>=t]
                print(t,"\t",np.mean(data.model_pos),"\t", np.median(data.model_pos),"\t",np.mean(data.baseline_pos),"\t", np.median(data.baseline_pos),"\t", len(data))

            print("-" * 100)

            #Para el número de reviews del usuario
            for t in n_items:
                data = RET.loc[RET.usr_train_reviews>=t]
                print(t,"\t",np.mean(data.model_pos),"\t", np.median(data.model_pos),"\t",np.mean(data.baseline_pos),"\t", np.median(data.baseline_pos),"\t", len(data))

            RET.to_excel("docs/"+self.DATE+"/"+self.MODEL_NAME+"_"+self.CITY+"_LIKE.xlsx") # Por usuario

    ####################################################################################################################

    def getFilteredData(self,verbose=True):

        def dropMultipleVisits(data):
            # Si un usuario fue multiples veces al mismo restaurante, quedarse siempre con la última (la de mayor reviewId)
            multiple = data.groupby(["userId", "restaurantId"])["reviewId"].max().reset_index(name="last_reviewId")
            return data.loc[data.reviewId.isin(multiple.last_reviewId.values)].reset_index(drop=True)

        ################################################################################################################

        IMG = pd.read_pickle(self.PATH + "img-option" + str(self.OPTION) + "-new.pkl")
        #IMG = pd.read_pickle(self.PATH + "img-food.pkl")
        RVW = pd.read_pickle(self.PATH + "reviews.pkl")

        RVW = RVW.drop(columns="index")

        IMG['review'] = IMG.review.astype(int)
        RVW["reviewId"] = RVW.reviewId.astype(int)

        RVW["num_images"] = RVW.images.apply(lambda x: len(x))
        RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)
        RVW = RVW.loc[(RVW.userId != "")]

        # Quedarse con ultima review de los usuarios en caso de tener valoraciones diferentes (mismo rest)
        # --------------------------------------------------------------------------------------------------------------

        RVW = dropMultipleVisits(RVW)

        # Eliminar usuarios con menos de n reviews DE LOS QUE NO TIENEN IMÁGEN
        # --------------------------------------------------------------------------------------------------------------


        if(self.CONFIG["min_revs_rst"]>0):

            self.printE("SE FILTRAN LAS REVIEWS DE LOS RESTAURANTES")
            DROP_RVWS = RVW.groupby("restaurantId")["like"].count().reset_index(name="revs")
            DROP_RVWS = DROP_RVWS.loc[(DROP_RVWS.revs < self.CONFIG["min_revs_rst"])].restaurantId.values
            DROP_RVWS = RVW.loc[RVW.restaurantId.isin(DROP_RVWS)].index.values

            RVW = RVW.loc[~RVW.index.isin(DROP_RVWS)].reset_index(drop=True)

        if (self.CONFIG["min_revs_usr"] > 0):

            self.printE("SE FILTRAN LAS REVIEWS DE LOS USUARIOS")

            DROP_RVWS = RVW.groupby("userId")["like"].count().reset_index(name="revs")
            DROP_RVWS = DROP_RVWS.loc[(DROP_RVWS.revs < self.CONFIG["min_revs_usr"])].userId.values
            DROP_RVWS = RVW.loc[RVW.userId.isin(DROP_RVWS)].index.values

            RVW = RVW.loc[~RVW.index.isin(DROP_RVWS)].reset_index(drop=True)

        #Todo: Mirar despues cuantas reviews quedan en los restaurantes


        # Obtener ID para ONE-HOT de usuarios y restaurantes
        # --------------------------------------------------------------------------------------------------------------

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

                if(ITEMS_PER_USR>0):

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
                ITEMS_PER_USR =  int((N_RST / N_USR) * 100)
                #ITEMS_PER_USR =  int((N_USR / N_RST) * 5)

            else: ITEMS_PER_USR = int(self.CONFIG["neg_likes"])

            if(not os.path.exists(file_path+"TRAIN_LIKE")):

                TRAIN_LIKE = self.getPickle(split_file_path_like, "TRAIN")
                #TRAIN_IMG = self.getPickle(split_file_path_take, "TRAIN")

                #TRAIN = TRAIN_LIKE.append(TRAIN_IMG, ignore_index=True)
                TRAIN = TRAIN_LIKE
                self.printE("NO SE AÑADEN LAS REVIEWS CON IMAGEN")

                TRAIN = compensateItems(TRAIN)

                TRAIN = utils.shuffle(TRAIN, random_state=self.SEED).reset_index(drop=True)
                self.toPickle(file_path, "TRAIN_LIKE", TRAIN);
                del TRAIN

            else: self.printW("Ya existen las preferencias de TRAIN_LIKE, se omite...")

            if (not os.path.exists(file_path + "TRAIN_DEV_LIKE")):

                TRAIN_DEV_LIKE = self.getPickle(split_file_path_like, "TRAIN_DEV")
                #TRAIN_DEV_IMG = self.getPickle(split_file_path_take, "TRAIN_DEV")

                #TRAIN_DEV = TRAIN_DEV_LIKE.append(TRAIN_DEV_IMG, ignore_index=True)
                TRAIN_DEV = TRAIN_DEV_LIKE
                self.printE("NO SE AÑADEN LAS REVIEWS CON IMAGEN")

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

                tmp = pd.concat([tmp] * len(data))

                tmp["is_dev"] = 0
                tmp["id_test"] = np.repeat(data.id_img.values,len(tmp)/len(data))

                data['is_dev'] = 1
                data['id_test'] = data.id_img

                tmp = tmp.append(data, ignore_index=True)

                tmp['id_user'] = id_u

                tmp = tmp[['id_user', 'id_restaurant', 'id_img', 'is_dev','id_test']]

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
                new_rests = random.sample(available_rests, k=new_items)

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
                #DEV_T = self.getPickle(split_file_path_take, "DEV")
                #DEV = DEV_L.append(DEV_T.loc[DEV_T.like==1],ignore_index=True)

                DEV = DEV_L
                self.printE("NO SE AÑADEN LAS REVIEWS CON IMAGEN")

                TRAIN = self.getPickle(file_path, "TRAIN_LIKE")

                DEV = DEV.groupby(["id_user", "id_restaurant"]).progress_apply(topNfn,prev_set=TRAIN, numb_rests=N_RESTS).reset_index(drop=True)
                self.toPickle(file_path, "DEV_LIKE", DEV); del DEV

            else: self.printW("Ya se ha creado el conjunto de DEV de likes, se omite...")

            if(not os.path.exists(file_path+"TEST_LIKE")):
                TEST_L = self.getPickle(split_file_path_like, "TEST")
                #TEST_T = self.getPickle(split_file_path_take, "TEST")
                #TEST = TEST_L.append(TEST_T.loc[TEST_T.like==1],ignore_index=True)

                TEST = TEST_L
                self.printE("NO SE AÑADEN LAS REVIEWS CON IMAGEN")

                TRAIN_DEV = self.getPickle(file_path, "TRAIN_DEV_LIKE")

                TEST = TEST.loc[TEST.like==1] #Solo los positivos para este caso (ESTO VALE PARA ALGO?)

                TEST = TEST.groupby(["id_user", "id_restaurant"]).progress_apply(topNfn,prev_set=TRAIN_DEV, numb_rests=N_RESTS).reset_index(drop=True)
                self.toPickle(file_path, "TEST_LIKE", TEST);del TEST

            else: self.printW("Ya se ha creado el conjunto de TEST de likes, se omite...")

            return

        ################################################################################################################

        # Mirar si ya existen los datos
        # --------------------------------------------------------------------------------------------------------------

        file_path = self.PATH + self.MODEL_NAME.upper()
        split_file_path_take = file_path + "/original_take/"
        split_file_path_like = file_path + "/original_like_["+str(self.CONFIG['min_revs_usr'])+"|"+str(self.CONFIG['min_revs_rst'])+"]/"

        file_path += "/data_%s_%s_[%d|%d]/" % (self.CONFIG['neg_images'],self.CONFIG["neg_likes"],self.CONFIG['min_revs_usr'],self.CONFIG['min_revs_rst'])

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

        # Separar reviews con y sin imágen.
        # ---------------------------------------------------------------------------------------------------------------

        RVW_IM = RVW.loc[(RVW.num_images > 0)]  # Reviews sin imagen
        RVW_IM = RVW_IM.drop(columns=["num_images"])

        RVW_NI = RVW.loc[(RVW.num_images == 0)]  # Reviews con imagen
        RVW_NI = RVW_NI.drop(columns=["num_images"])

        # Mover ejemplos positivos a donde corresponde (N, 1, 1)
        # --------------------------------------------------------------------------------------------------------------

        splitIMGSets(split_file_path_take, RVW_IM)
        #splitLIKESets(split_file_path_like, RVW_NI) #Como el anterior, pero solo mueve likes a DEV y TEST

        self.printE("UTILIZAR SOLO LOS QUE NO TIENEN IMÁGEN!!") # Si no, al hacer BOTH, puede haber datos de DEV de IMG en TRAIN de esto.
        splitLIKESets(split_file_path_like, RVW)

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

