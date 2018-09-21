# -*- coding: utf-8 -*-import urllib.requestimport pandas as pdimport reimport jsonimport osimport sslimport timefrom TripAdvisor import *import requestsfrom os import listdirfrom os.path import isfile, joinimport numpy as np########################################################################################################################def waitForEnd(threads):    stop=False    while not stop:        stop = True        for i in threads:            if(i.isAlive()):stop=False        time.sleep(5)    print("END")    print("-"*50)def stepOne(CITY):    TAH = TripAdvisorHelper()    PAGES = TAH.getRestaurantPages(CITY)    n_threads = 20    threads = []    len_data = PAGES    len_data_thread = len_data // n_threads    for i in range(n_threads):        data_from = i * len_data_thread        data_to = (i + 1) * len_data_thread        if (i == n_threads - 1): data_to = len_data        temp_thread = TripAdvisor(i, "Thread-" + str(i), i, data=[data_from, data_to], city=CITY, step=0)        threads.append(temp_thread)        threads[i].start()    waitForEnd(threads)    TAH.joinRestaurants()def stepTwo():    TAH = TripAdvisorHelper()    n_threads = 25    threads = []    data = pd.read_pickle("restaurants.pkl")    len_data = len(data)    len_data_thread = len_data // n_threads    for i in range(n_threads):        data_from = i * len_data_thread        data_to = (i + 1) * len_data_thread        if (i == n_threads - 1): data_to = len_data        data_thread = data.iloc[data_from:data_to, :].reset_index()        temp_thread = TripAdvisor(i, "Thread-" + str(i), i, data=data_thread, step=1)        threads.append(temp_thread)        threads[i].start()    waitForEnd(threads)    TAH.joinReviews()def stepThree():    TAH = TripAdvisorHelper()    n_threads = 20    threads = []    data = pd.read_pickle("reviews.pkl")    data = data.loc[(data.title == '') | (data.text.isnull())]  # Si no tienen titulo o texto    len_data = len(data)    len_data_thread = len_data//n_threads    for i in range(n_threads):        data_from = i*len_data_thread        data_to = (i+1)*len_data_thread        if(i==n_threads-1):data_to = len_data        data_thread = data.iloc[data_from:data_to,:].reset_index()        temp_thread = TripAdvisor(i, "Thread-"+str(i), i, data=data_thread, step=2)        threads.append(temp_thread)        threads[i].start()    waitForEnd(threads)    TAH.joinAndAppendFiles()def stepFour():    n_threads = 25    threads = []    data = pd.read_pickle("reviews.pkl")    len_data = len(data)    len_data_thread = len_data // n_threads    for i in range(n_threads):        data_from = i * len_data_thread        data_to = (i + 1) * len_data_thread        if (i == n_threads - 1): data_to = len_data        data_thread = data.iloc[data_from:data_to, :].reset_index()        temp_thread = TripAdvisor(i, "Thread-" + str(i), i, data=data_thread, step=3)        threads.append(temp_thread)        threads[i].start()def getStats(CITY):    FOLDER = "../"+CITY.lower()+"_data/"    IMG_FOLDER = FOLDER + "images/"    RST = pd.read_pickle(FOLDER + "restaurants.pkl")    USRS = pd.read_pickle(FOLDER + "users.pkl")    RVW = pd.read_pickle(FOLDER + "reviews.pkl")    # Añadir columnas con número de imágenes y likes    RVW["num_images"] = RVW.images.apply(lambda x: len(x))    RVW["like"] = RVW.rating.apply(lambda x: 1 if x > 30 else 0)    RVW["restaurantId"] = RVW.restaurantId.astype(int)    #Añadir columnas a los restaurantes    RST["id"] = RST.id.astype(int)    RST["reviews"] = 0    print("Retaurantes: " + str(len(RST.id.unique())))    print("Usuarios: " + str(len(USRS.loc[(USRS.id != "")])))    print("Reviews: " + str(len(RVW.loc[(RVW.userId != "")])))    print("Imágenes: " + str(sum(RVW.num_images)))    print("")    # Quedarse con las que tienen imágenes    RVW = RVW.loc[RVW.num_images > 0]    # Para cada restaurante    #for i, r in RVW.groupby("restaurantId"):    #    likes = (sum(r.like) * 100) / len(r)    #    RST.loc[RST.id == i, ["like_prop", "reviews"]] = likes, len(r)    # Quedarse con los que tienen reviews con imágen    RST = RST.loc[RST.reviews > 0]    print("Restaurantes (con imágen): ", len(RST))    print("Usuarios (con imágen): ", len(RVW.userId.unique()))    print("Reviews (con imágen): ", len(RVW))    print("")    print("Porcentaje de likes: ", sum(RVW.like)/len(RVW)*100)    RET = pd.DataFrame(columns=['user','likes','reviews'])    for i, g in RVW.groupby("userId"):        likes = sum(g.like)        total = int(len(g))        RET = RET.append({"user":i,"likes":likes,"reviews":total}, ignore_index=True)    RET.to_csv("../../stats/user_stats_"+CITY.lower()+".csv")#-----------------------------------------------------------------------------------------------------------------------def init():    #Descargar restaurantes    #--------------------------------------------------------------------------    CITY = "Oviedo"    #stepOne(CITY)    #Descargar reviews    #--------------------------------------------------------------------------    #stepTwo()    #Ampliar reviews    #--------------------------------------------------------------------------    #stepThree()    #Descargar imagenes    #--------------------------------------------------------------------------    #stepFour()    #Obtener estadisticas    getStats(CITY)    # TODO: USERS SIN ID, MISMO RESTAURANTE, 2 IDif __name__ == "__main__":    init()########################################################################################################