# -*- coding: utf-8 -*-

import pandas as pd
import threading
import time
import requests
from time import sleep

from tqdm import tqdm
import ssl
from pyquery import PyQuery
import urllib.request
import urllib
import re
import os
import json

class TripAdvisorHelper():

    def __init__(self):
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def joinReviews(self):

        revs = pd.DataFrame(columns=TripAdvisor.review_cols)
        usrs = pd.DataFrame(columns=TripAdvisor.user_cols)

        for f in os.listdir(TripAdvisor.TMP_FOLDER):

            if(("reviews" in f) and (".pkl" in f)):
                tmp_rev = pd.read_pickle(TripAdvisor.TMP_FOLDER+"/"+f)
                revs = revs.append(tmp_rev,ignore_index=True)

            if(("users" in f) and (".pkl" in f)):
                tmp_usr = pd.read_pickle(TripAdvisor.TMP_FOLDER+"/"+f)
                usrs = usrs.append(tmp_usr,ignore_index=True)

        usrs = usrs.drop_duplicates("id")
        revs = revs.drop_duplicates("reviewId")

        if(len(usrs)>0):pd.to_pickle(usrs,"users.pkl")
        if(len(revs)>0):pd.to_pickle(revs,"reviews.pkl")

        #Eliminar los ficheros de la carpeta.
        for f in os.listdir(TripAdvisor.TMP_FOLDER):
            if((("reviews-" in f) or ("users-" in f)) and (".pkl" in f)):
                os.remove(TripAdvisor.TMP_FOLDER+"/"+f)

    def joinAndAppendFiles(self):

        RV = pd.read_pickle("reviews.pkl")

        revs = pd.DataFrame(columns=TripAdvisor.review_cols)

        for f in os.listdir(TripAdvisor.TMP_FOLDER):

            if (("reviews" in f) and (".pkl" in f)):
                tmp_rev = pd.read_pickle(TripAdvisor.TMP_FOLDER + "/" + f)
                revs = revs.append(tmp_rev, ignore_index=True)

        RET  = RV.loc[(~RV.reviewId.isin(revs.reviewId.values))]
        RET = RET.append(revs, ignore_index=True)

        #Reemplazar fichero de reviews
        if(len(RET)>0):
            pd.to_pickle(RET,"reviews.pkl")

        # Eliminar los ficheros de la carpeta.
        for f in os.listdir(TripAdvisor.TMP_FOLDER):
            if((".pkl" in f) and(("reviews-" in f) or ("users-" in f))):
                os.remove(TripAdvisor.TMP_FOLDER + "/" + f)

    def joinRestaurants(self):

        rest = pd.DataFrame(columns=TripAdvisor.rest_cols)

        for f in os.listdir(TripAdvisor.TMP_FOLDER):

            if (("restaurants-" in f) and (".pkl" in f)):
                tmp_rest = pd.read_pickle(TripAdvisor.TMP_FOLDER + "/" + f)
                rest = rest.append(tmp_rest, ignore_index=True)

        if (len(rest) > 0): pd.to_pickle(rest, "restaurants.pkl")

        # Eliminar los ficheros de la carpeta.
        for f in os.listdir(TripAdvisor.TMP_FOLDER):
            if (("restaurants-" in f) and (".pkl" in f)):
                os.remove(TripAdvisor.TMP_FOLDER + "/" + f)

    def getRestaurantPages(self,CITY):
        url = "https://www.tripadvisor.es/RestaurantSearch?Action=PAGE&geo=" + str(self.getGeoId(CITY)) + "&sortOrder=alphabetical"
        params = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36"}
        r = requests.get(url, headers=params)
        pq = PyQuery(r.text)
        n = [i.text() for i in pq.items('a.pageNum.taLnk')]

        if(len(n)>0):n= n[-1]
        else:n=1

        return(int(n))

    def getGeoId(self, CITY):
        url = "https://www.tripadvisor.es/TypeAheadJson?action=API&query=" + CITY
        params = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36"}

        r = requests.get(url, headers=params)
        response = json.loads(r.text)
        id = int(response['results'][0]['value'])

        return id


class TripAdvisor(threading.Thread):

    TMP_FOLDER = "temp_data"
    BASE_URL = "https://www.tripadvisor.es"
    RESTAURANTS_URL = BASE_URL+"/Restaurants"
    GEO_ID = 0
    SUCCESS_TAG = 200

    CITY = None
    DATA = None
    STEP = None
    ITEMS = None

    rest_cols = ["id","name","city","priceInterval","url","rating"]
    review_cols = ["reviewId", "userId", "restaurantId", "title", "text", "date", "rating", "language", "images", "url"]
    user_cols = ["id", "name", "location"]

    def __init__(self, threadID, name, counter,city = "Barcelona", data = None, step=1):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

        self.CITY = city
        self.PATH = "/media/HDD/pperez/TripAdvisor/" + city.lower() + "_data/"

        self.GEO_ID = TripAdvisorHelper().getGeoId(CITY=city)
        self.DATA = data
        self.STEP = step
        if(data is None):self.ITEMS = len(data)

        pd.set_option('display.max_rows', 10)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

    def run(self):
        print("Starting " + self.name)

        #Descargar lista de restaurantes...
        if(self.STEP==0):
            self.downloadRestaurants();
        #Descargar reviews...
        elif(self.STEP==1):
            self.downloadReviewData()

        #Completar reviews...
        elif(self.STEP==2):
            self.completeReviews()

        #Descargar imágenes...
        elif(self.STEP==3):
            self.downloadImages()

        print("Exiting " + self.name)

    #-------------------------------------------------------------------------------------------------------------------

    def downloadRestaurants(self):

        def getPage(page):
            items_page = 30

            url = "https://www.tripadvisor.es/RestaurantSearch?Action=PAGE&geo=" + str(self.GEO_ID) + "&sortOrder=alphabetical&o=a" + str((page) * items_page)
            params = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36"}

            r = requests.get(url, headers=params)
            return PyQuery(r.text)

        #---------------------------------------------------------------------------------------------------------------

        data = pd.DataFrame(columns=self.rest_cols)

        fromPage= self.DATA[0]
        toPage = self.DATA[1]

        for p in range(fromPage,toPage):
            print("Thread "+str(self.threadID)+": "+str(p+1)+" de "+str(toPage))

            pq = getPage(p)
            rst_in_pg = pq("div#EATERY_SEARCH_RESULTS")

            for r in rst_in_pg.items("div.listing.rebrand"):
                id = r.attr("id").replace("eatery_","")
                name = r("div.title>a").text()
                url = self.BASE_URL+r("div.title>a").attr("href")
                rating = r("span.ui_bubble_rating").attr("class")
                if(rating!=None):rating = int(rating.split(" ")[-1].replace("bubble_",""))
                price = r("div.cuisines>span.item.price").text()

                data = data.append({"id":id,"name":name,"city":self.CITY,"priceInterval":price,"url":url,"rating":rating},ignore_index=True)

        pd.to_pickle(data, self.TMP_FOLDER + "/restaurants-" + str(self.threadID) + ".pkl")

    #-------------------------------------------------------------------------------------------------------------------

    def completeReviews(self):

        def xpanReviews(RV):

            ROWS = pd.DataFrame(columns=['reviewId','title','text'])

            ids = RV.reviewId.values
            rev_str = ','.join([str(x) for x in ids])

            data = {'reviews': rev_str, 'widgetChoice': 'EXPANDED_HOTEL_REVIEW_HSX', 'Action': 'install'}
            params = {'Content-Type': 'application/x-www-form-urlencoded'}
            r = requests.post("https://www.tripadvisor.es/OverlayWidgetAjax?Mode=EXPANDED_HOTEL_REVIEWS&metaReferer=Restaurant_Review",data=data, headers=params)

            pq = PyQuery(r.text)
            itms = pq.items("div[data-reviewlistingid]")


            for i in itms:
                revID = str(i.attr('data-reviewlistingid'))
                title = i.find('a#rn' + revID + ">span").text()
                text = i.find('p.partial_entry').text()

                if(title==""):
                    print(RV.loc[(RV.reviewId == revID)].url.values)
                    print("No_TITTLE")

                ROWS = ROWS.append({'reviewId':revID,'title':title,'text':text},ignore_index=True)

            RV = RV.drop(columns=["title","text"])
            RET = RV.merge(ROWS,left_on='reviewId', right_on='reviewId', how='inner')

            return RET

        #---------------------------------------------------------

        revs = pd.DataFrame(columns=self.review_cols)

        RV = self.DATA

        total = len(RV)
        per_post = 500
        its = total // per_post

        allD = 0

        for i in range(its):
            print("Thread "+str(self.threadID)+": "+str(i+1)+" de "+str(its))

            data_from = i * per_post
            data_to = (i + 1) * per_post
            if(i==its-1): data_to = total

            temp_data = RV.iloc[data_from:data_to,:]
            new_data = xpanReviews(temp_data)

            if(len(temp_data)!=len(new_data)):
                print(list(set(temp_data.reviewId.values)-set(new_data.reviewId.values)))
            revs = revs.append(new_data,ignore_index=True)


        pd.to_pickle(revs, self.TMP_FOLDER + "/reviews-" + str(self.threadID) + ".pkl")

    #-------------------------------------------------------------------------------------------------------------------

    def downloadImages(self):

        def saveImage(path, img_src):

            # si está descargada, skip
            if (os.path.isfile(path)): return True

            gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # Only for gangstars

            try:
                a = urllib.request.urlopen(img_src, context=gcontext)
            except:
                return False

            if (a.getcode() != self.SUCCESS_TAG):
                return False

            try:
                f = open(path, 'wb')
                f.write(a.read())
                f.close()
                return path
            except Exception as e:
                print(e)
                return False

        def download(revId, rev_url, images, highRes=False):

            ret = images
            item = 0

            if(highRes):path = self.PATH+"images/" + str(revId)
            else:path = self.PATH+"images_lowres/" + str(revId)

            if not os.path.exists(path): os.makedirs(path)

            for i in ret:
                name = path + "/" + str(item) + ".jpg"
                url_high_res = i['image_url_lowres']

                # Cambiar la URL de la imagen low-res a la high-res

                if(highRes):
                    url_high_res = url_high_res.replace("/photo-l/", "/photo-o/")
                    url_high_res = url_high_res.replace("/photo-f/", "/photo-o/")

                    saved = saveImage(name, url_high_res)
                    if (not saved):
                        # Algunas veces hay que cambiarlo por photo-w
                        url_high_res = url_high_res.replace("/photo-o/", "/photo-w/")
                        saved = saveImage(name, url_high_res)
                        if (not saved):
                            # Algunas veces hay que cambiarlo por photo-p
                            url_high_res = url_high_res.replace("/photo-w/", "/photo-p/")
                            saved = saveImage(name, url_high_res)
                            if (not saved):
                                # Algunas veces hay que cambiarlo por photo-s
                                url_high_res = url_high_res.replace("/photo-p/", "/photo-s/")
                                saved = saveImage(name, url_high_res)
                                if (not saved): print("\nImg not saved: " + str(url_high_res) + " " + str(rev_url))

                else:
                    saved = saveImage(name, url_high_res)
                    if (not saved):
                        print("Error-"+str(url_high_res)+"-"+name)

                i['image_path'] = name
                i['image_high_res'] = url_high_res

                item += 1

            return ret
            # ------------------------------------------------------------

        RV = self.DATA

        # Para cada una de las reviews...
        for i, r in RV.iterrows():
            print("Thread "+str(self.threadID)+": "+str(i+1)+" de "+str(len(RV)))
            imgs = r.images
            revId = r.reviewId

            # Si tiene imagenes...
            if (len(imgs) > 0):
                r.images = download(revId, r.url, imgs)

    #-------------------------------------------------------------------------------------------------------------------

    def downloadReviewData(self,maxRest=20):

        restaurants = self.DATA

        items = 0

        revs = pd.DataFrame(columns=self.review_cols)
        usrs = pd.DataFrame(columns=self.user_cols)

        for i, r in restaurants.iterrows():

            print("Thread "+str(self.threadID)+": "+str(i+1)+" de "+str(len(restaurants)))
            rest_data = r
            rest_id = r.id

            items += 1

            #Si no hay reviews se salta
            try:res_hmtl = PyQuery(self.getHtml(r.url))
            except: continue

            #Si no hay reviews (en español) se salta
            pg_revs = res_hmtl.find("div.reviewSelector")
            if(len(pg_revs)==0):continue

            #Obtener el número de revs
            total_num = res_hmtl.find("label[for='taplc_location_review_filter_controls_0_filterLang_es']>span").text().replace(")","").replace("(","").replace(".","")
            total_num = int(total_num)

            us, rv = self.getReviews(res_hmtl, rest_id, rest_data,total_num)

            revs = revs.append(rv, ignore_index=True)
            usrs = usrs.append(us, ignore_index=True)


            # Si se alcanza un máximo o es el final, guardar.
            if (items == maxRest or i == len(restaurants) - 1):
                if(i==len(restaurants)-1):print("Last saving...")
                else:print("Saving...")

                self.appendPickle(revs, self.TMP_FOLDER+"/reviews-"+str(self.threadID)+".pkl")
                self.appendPickle(usrs, self.TMP_FOLDER+"/users-"+str(self.threadID)+".pkl")

                del revs, usrs

                revs = pd.DataFrame(columns=self.review_cols)
                usrs = pd.DataFrame(columns=self.user_cols)

                items = 0

    def getReviews(self, pq, rest_id, rest_data,total_num):

        revs = pd.DataFrame(columns=self.review_cols)
        usrs = pd.DataFrame(columns=self.user_cols)


        # Si existe el boton, hay más páginas
        if (total_num>10):

            i = 1

            temp_url = rest_data['url']

            pages = total_num//10
            if(total_num%10>0):pages+=1

            for p in range(pages):

                usr_data, rev_data, cont = self.parseReviewPage(pq, temp_url, rest_id, rest_data)

                revs = revs.append(rev_data)
                usrs = usrs.append(usr_data)

                #Si se cortó en medio de la página, son traducciones
                if (len(revs)<10): break

                temp_url = rest_data['url'].replace("-Reviews-", "-Reviews-or" + str(i * 10) + "-")
                pq = PyQuery(self.getHtml(temp_url))
                i += 1

        else:

            usrs, revs, cont = self.parseReviewPage(pq, rest_data['url'], rest_id, rest_data)

        return usrs, revs

    def getReviewsOLD(self,pq, rest_id, rest_data):

        revs = pd.DataFrame(columns=self.review_cols)
        usrs = pd.DataFrame(columns=self.user_cols)

        buttonNextPage = pq("a.nav.next.taLnk.ui_button.primary")

        # Si existe el boton, hay más páginas
        if (buttonNextPage):

            i = 1

            temp_url = rest_data['url']

            while True:

                usr_data, rev_data, cont = self.parseReviewPage(pq, temp_url, rest_id, rest_data)

                revs = revs.append(rev_data)
                usrs = usrs.append(usr_data)

                if (pq("a.nav.next.ui_button.primary.disabled") or cont == 0): break

                temp_url = rest_data['url'].replace("-Reviews-", "-Reviews-or" + str(i * 10) + "-")

                pq = PyQuery(self.getHtml(temp_url))

                i += 1

        else:

            usrs, revs, cont = self.parseReviewPage(pq, rest_data['url'], rest_id, rest_data)

        print(len(revs))

        return usrs, revs

    def parseReviewPage(self,pq, page_url, rest_id, rest_data):

        rev_data = pd.DataFrame(columns=self.review_cols)
        usr_data = pd.DataFrame(columns=self.user_cols)

        for rev in pq.items("div.review-container"):

            # Si aparece el banner, no continuar, las demás son traducidas
            if (rev("div.translationOptions")):
                return usr_data, rev_data, 0

            rev_id = rev("div.reviewSelector").attr("data-reviewid")
            rev_title = rev("span.noQuotes").text()
            rev_url = self.BASE_URL + str(rev("div.quote>a").attr("href"))
            rev_date = rev("span.ratingDate.relativeDate").attr("title")
            rev_rating = int(
                rev("span.ui_bubble_rating").remove_class("ui_bubble_rating").attr("class").replace("bubble_", ""))
            rev_content = None
            rev_images = []

            user_id = str(rev("div.avatar").remove_class("avatar").attr("class"))[8:]
            user_name = rev("span.expand_inline.scrname").text()
            user_loc = rev("span.expand_inline.userLocationd").text()

            # Ver si se puede expandir
            more = rev("div.prw_rup.prw_reviews_text_summary_hsx>div.entry>p.partial_entry>span.taLnk.ulBlueLinks")

            # Si no se puede expandir, obtener el texto
            if (not more):
                rev_content = rev("div.prw_rup.prw_reviews_text_summary_hsx>div.entry>p.partial_entry").text()

            # Si hay imagenes, obtener los links
            images = rev("div.inlinePhotosWrapper")

            if (images):
                rev_images = self.getImages(rev)

            # addUser(userId=user_id,username=user_name,location=user_loc)
            # addReview(reviewId=rev_id,userId=user_id,restaurantId=rest_id,title=rev_title, text=rev_content,date=rev_date,rating=rev_rating,url=rev_url,language="es",images=rev_images)

            rev_content = {'reviewId': rev_id, 'userId': str(user_id), 'restaurantId': str(rest_id), 'title': rev_title,
                           'text': rev_content, 'date': rev_date, 'rating': rev_rating, 'language': "es",
                           'images': rev_images, 'url': rev_url}
            rev_data = rev_data.append(rev_content, ignore_index=True)

            usr_content = {'id': user_id, 'name': user_name, 'location': user_loc}
            usr_data = usr_data.append(usr_content, ignore_index=True)

        return usr_data, rev_data, 1

    def getImages(self,dt):

            ret = []

            for img in dt.items("noscript>img.centeredImg.noscript"):
                ret.append({"image_url_lowres": img.attr("src"), "image_path": "", "image_high_res": ""})
            return ret

    def appendPickle(self,itm, name):

        if os.path.isfile(name):
            file = pd.read_pickle(name)
            file = file.append(itm, ignore_index=True)
            pd.to_pickle(file, name)
        else:
            pd.to_pickle(itm, name)

    def getHtml(self,url):

        '''
        gcontext = ssl.SSLContext(ssl.PROTOCOL_TLSv1)  # Only for gangstars

        req = urllib.request.Request(
            url,
            data=None,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
            }
        )

        fp = urllib.request.urlopen(req, context=gcontext)
        html = fp.read().decode("utf8")
        fp.close()
        '''

        data = {'filterLang': 'es'}
        params = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest"}

        #Reintentar la petición mientras no se obtenga resultado

        r = ''
        while r == '':
            try:
                r = requests.post(url, data=data, headers=params)
                break
            except:
                time.sleep(5)
                continue


        html = r.text

        r.close()

        return html

