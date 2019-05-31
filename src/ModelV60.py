# -*- coding: utf-8 -*-
from src.ModelV6 import *


########################################################################################################################
class DotBias(keras.layers.Layer):

    def __init__(self, use_bias=False, **kwargs):
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.Constant(value=0.0)
        super(DotBias, self).__init__(**kwargs)

    def build(self, input_shape):
        if (self.use_bias):
            self.bias = self.add_weight(shape=(1,), initializer=self.bias_initializer,name='bias', trainable=True)
            super(DotBias, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dot = K.batch_dot(x[0], x[1], axes=1)
        if (self.use_bias):
            return K.bias_add(dot, self.bias)
        else:
            return dot

    def compute_output_shape(self, input_shape):
        return (None, 1)

class ModelV60(ModelV6):

    def __init__(self,city,config,date,seed = 2):

        ModelV6.__init__(self,city,config,date,seed = seed)

        self.MODEL_NAME = "modelv60"

        if(self.CONFIG["use_images"]):
            if(self.CONFIG["use_like"]):submodel = "both"
            else: submodel = "take"
        elif(self.CONFIG["use_like"]): submodel = "like"
        else: submodel = "none"

        self.MODEL_PATH = "models/"+self.MODEL_NAME+"/" + self.CITY.lower()+"_"+submodel

        print(" " + self.MODEL_NAME + " Básico")
        print("#" * 50)

    ####################################################################################################################

    def getModel(self):

        # Fijar las semillas de numpy y TF
        np.random.seed(self.SEED)
        rn.seed(self.SEED)
        tf.set_random_seed(self.SEED)


        emb_size = 256
        activation_fn = "relu"
        
        usr_emb_size = emb_size
        rst_emb_size = emb_size
        img_emb_size = emb_size

        #init = keras.initializers.RandomUniform(minval=0, maxval=0.1, seed=None)

        model_u = Sequential()
        model_u.add(Embedding(self.DATA["N_USR"], usr_emb_size, input_shape=(1,), name="in_usr",  embeddings_constraint=keras.constraints.nonneg()))
        model_u.add(Flatten())

        model_r = Sequential()
        model_r.add(Embedding(self.DATA["N_RST"], rst_emb_size, input_shape=(1,), name="in_rst",  embeddings_constraint=keras.constraints.nonneg()))
        model_r.add(Flatten())

        model_i = Sequential()
        model_i.add(Dense(img_emb_size, input_shape=(self.DATA["V_IMG"],), name="in_img", use_bias=True))

        #  LIKE --------------------------------------------------------------------------------------------------------

        conc_like_out = Dot(axes=1)([model_u.output, model_r.output])

        #conc_like_out = DotBias(use_bias=True)([model_u.output, model_r.output])

        #  TAKE --------------------------------------------------------------------------------------------------------

        conc_take_out = Dot(axes=1)([model_u.output, model_i.output])

        opt = Adam(lr=self.CONFIG["learning_rate"])

        model_like = Model(inputs=[model_u.input, model_r.input], outputs=conc_like_out)
        model_like.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        model_take = Model(inputs=[model_u.input, model_i.input], outputs=conc_take_out)
        model_take.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        plot_model(model_take,"model_take.png")
        plot_model(model_like,"model_like.png")

        return (model_like, model_take)

        #if (self.CONFIG["use_images"] == 1): return (model_like,model_take)
        #else: return model_like

    def train_batches(self, sq_take, sq_like):

        def _train_random_batch(model,data):

            bn =np.random.randint(data.__len__())
            ret = model.fit(data.__getitem__(bn)[0],data.__getitem__(bn)[1],verbose=0)

            return ret

        #---------------------------------------------------------------------------------------------------------------

        if ("linear_cosine" in self.CONFIG["lr_decay"]):
            val = K.get_value(tf.train.linear_cosine_decay(self.CONFIG["learning_rate"], self.CURRENT_EPOCH, self.CONFIG["epochs"]))

            if(self.CONFIG["use_images"]==1):
                K.set_value(self.MODEL[0].optimizer.lr, val)
                K.set_value(self.MODEL[1].optimizer.lr, val)
            else:
                K.set_value(self.MODEL.optimizer.lr, val)


        for b in range(max(sq_take.__len__(), sq_like.__len__())):

            if(self.CONFIG["use_images"]==1):
                # Entrenar 2 partes
                like_ret = _train_random_batch(self.MODEL[0], sq_like)
                take_ret = _train_random_batch(self.MODEL[1], sq_take)

                #np.sum(self.MODEL[0].get_layer("in_usr").get_weights()[0] - self.MODEL[1].get_layer("in_usr").get_weights()[0])
                #return (0,0, take_ret.history['loss'][0], take_ret.history['acc'][0])

            else:
                # Entrenar solo 1
                like_ret = _train_random_batch(self.MODEL, sq_like)


        if (self.CONFIG["use_images"] == 1): return [like_ret.history['loss'][0], take_ret.history['loss'][0]]
        else: return [like_ret.history['loss'][0]]

    def train(self, sq_take, sq_like):

        def _train(model,data):

            ret = model.fit_generator(data,
                                      epochs=1,
                                      steps_per_epoch=data.__len__(),
                                      shuffle=True,
                                      use_multiprocessing=False,
                                      workers=20,
                                      verbose=0)

            return ret

        #---------------------------------------------------------------------------------------------------------------

        ret={}


        if ("linear_cosine" in self.CONFIG["lr_decay"]):
            val = K.get_value(tf.train.linear_cosine_decay(self.CONFIG["learning_rate"], self.CURRENT_EPOCH, self.CONFIG["epochs"]))
            K.set_value(self.MODEL[0].optimizer.lr, val)
            K.set_value(self.MODEL[1].optimizer.lr, val)
            ret["LDECAY"] = val/self.CONFIG["learning_rate"]

        if(self.CONFIG["use_like"]==1):
            like_ret = _train(self.MODEL[0], sq_like)
            ret["L_LOSS"] = like_ret.history['loss'][0]

        if (self.CONFIG["use_images"] == 1):
            take_ret = _train(self.MODEL[1], sq_take)
            ret["T_LOSS"] = take_ret.history['loss'][0]

        return ret

    def dev(self, sq_take, sq_like):

        def _dev(model,data):
            ret = model.predict_generator(data, steps=data.__len__(), use_multiprocessing=False)
            tmp_like = data.DATA
            tmp_like["prediction"] = ret[:, 0]

            return tmp_like

        #---------------------------------------------------------------------------------------------------------------

        ret={}

        if(self.CONFIG["use_like"]==1):
            like_ret = _dev(self.MODEL[0], sq_like)
            mean_pos, median_pos = self.getTopN(like_ret)
            ret["L_AVG"] = mean_pos
            ret["L_MDN"] = median_pos


        if(self.CONFIG["use_images"]==1):
            take_ret = _dev(self.MODEL[1], sq_take)
            _,pcnt1_model,pcnt1_model_median  = self.getImagePos(take_ret)
            ret["T_AVG"] = pcnt1_model
            ret["T_MDN"] = pcnt1_model_median

        return ret

    ####################################################################################################################

    class TrainSequenceTake(Sequence):

        def __init__(self, data, batch_size, model):
            self.DATA = data
            self.MODEL = model
            self.BATCHES = np.array_split(data, len(data) // batch_size)
            self.BATCH_SIZE = batch_size

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):
            data_ids = self.BATCHES[idx]

            imgs = self.MODEL.DATA["IMG"][data_ids.id_img.values]

            #return ([np.array(data_ids.id_user.values), np.array(data_ids.id_restaurant.values),imgs],
            #        [np.array(data_ids[["take"]].values)])

            return ([np.array(data_ids.id_user.values), imgs],
                   [np.array(data_ids[["take"]].values)])

    class TrainSequenceLike(Sequence):

        def __init__(self, data, batch_size, model):
            self.DATA = data
            self.MODEL = model
            self.BATCHES = np.array_split(data, len(data) // batch_size)
            self.BATCH_SIZE = batch_size

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):
            data_ids = self.BATCHES[idx]

            return ([np.array(data_ids.id_user.values),
                     np.array(data_ids.id_restaurant.values)],
                    [np.array(data_ids[["like"]].values)])

    class DevSequenceTake(Sequence):

        def __init__(self, data, batch_size, model):
            self.DATA = data
            self.MODEL = model
            self.BATCHES = np.array_split(data, len(data) // batch_size)
            self.BATCH_SIZE = batch_size

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):
            data_ids = self.BATCHES[idx]

            imgs = self.MODEL.DATA["IMG"][data_ids.id_img.values]
            #return ([np.array(data_ids.id_user.values), np.array(data_ids.id_restaurant.values),imgs])
            return ([np.array(data_ids.id_user.values),imgs])

    class DevSequenceLike(Sequence):

        def __init__(self, data, batch_size, model):
            self.DATA = data
            self.MODEL = model
            self.BATCHES = np.array_split(data, len(data) // batch_size)
            self.BATCH_SIZE = batch_size

        def __len__(self):
            return len(self.BATCHES)

        def __getitem__(self, idx):
            data_ids = self.BATCHES[idx]

            return ([np.array(data_ids.id_user.values), np.array(data_ids.id_restaurant.values)])
