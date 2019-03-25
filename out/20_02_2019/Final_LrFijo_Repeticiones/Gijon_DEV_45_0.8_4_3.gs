Using TensorFlow backend.
[94mObteniendo datos...[0m
[93m[AVISO] 	Usuarios: 5139[0m
[93m[AVISO] 	Restaurantes: 598[0m
[93m[AVISO] Cargando datos generados previamente...[0m
[94mCreando modelo...[0m


##################################################
 MODELV4
##################################################
 modelv4d2
##################################################
[93m[AVISO] Existen 1 combinaciones posibles[0m
--------------------------------------------------
15d692719045548b880be3c9109b1159
--------------------------------------------------
[1mlearning_rate: [0m0.0001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	1.0017	27.1893	0.3198	0.2802
2	0.9403	26.4416	0.2984	0.2588
3	0.7232	24.9766	0.2826	0.243
4	0.534	24.6776	0.2737	0.2341
5	0.3744	25.5421	0.27	0.2304
6	0.2747	24.6846	0.2689	0.2294
7	0.2158	25.8107	0.2719	0.2324
8	0.1747	24.972	0.2684	0.2288
9	0.1465	24.3458	0.2675	0.2279
10	0.1238	25.4206	0.2703	0.2307
11	0.1079	25.4556	0.2683	0.2288
12	0.0936	24.2734	0.2695	0.23
13	0.0837	25.1939	0.2687	0.2291
14	0.0749	23.8621	0.2624	0.2228
15	0.0668	23.4486	0.2604	0.2209
16	0.0615	24.0164	0.2716	0.2321
17	0.0558	23.8364	0.2672	0.2276
18	0.0519	22.9509	0.2629	0.2233
19	0.0478	23.5748	0.2599	0.2203
20	0.044	23.5701	0.2555	0.2159
21	0.0416	24.2874	0.2674	0.2278
22	0.0382	23.75	0.2627	0.2231
23	0.0364	22.6495	0.2574	0.2178
24	0.0335	23.2383	0.2622	0.2226
25	0.0324	23.6986	0.2649	0.2253
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:833: DeprecationWarning: invalid escape sequence \w
  if re.match('\w:', url) or re.match(r'\\', url):
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:962: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', token) or re.search('\s$', token):
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:962: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', token) or re.search('\s$', token):
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:3610: DeprecationWarning: invalid escape sequence \|
  elif re.match('(or|\|\|)', conditional):
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:3840: DeprecationWarning: invalid escape sequence \.
  name = re.sub('\..*$', '', name)
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:5136: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', string) or re.search('\s$', string):
/usr/lib/python3/dist-packages/xlsxwriter/worksheet.py:5136: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', string) or re.search('\s$', string):
/usr/lib/python3/dist-packages/xlsxwriter/sharedstrings.py:103: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', string) or re.search('\s$', string):
/usr/lib/python3/dist-packages/xlsxwriter/sharedstrings.py:103: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', string) or re.search('\s$', string):
/usr/lib/python3/dist-packages/xlsxwriter/comments.py:160: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', text) or re.search('\s$', text):
/usr/lib/python3/dist-packages/xlsxwriter/comments.py:160: DeprecationWarning: invalid escape sequence \s
  if re.search('^\s', text) or re.search('\s$', text):
----------------------------------------------------------------------------------------------------
N_FOTOS_TRAIN (>=)	N_ITEMS	RND-MOD AC	CNT-MOD AC
9	186	0.11028662251900728	0.18225359376707723
5	339	0.08378743741585333	0.187950167082662
4	376	0.07211563471540244	0.1757927598370924
2	428	0.06383551743846039	0.16207309824019062
1	428	0.06383551743846039	0.16207309824019062
----------------------------------------------------------------------------------------------------
26	0.0303	22.2687	0.2592	0.2197
27	0.0292	22.4276	0.257	0.2174
28	0.0282	23.9486	0.2613	0.2217
29	0.0264	23.3598	0.2617	0.2222
30	0.0256	23.5	0.2669	0.2273
31	0.0242	23.1636	0.2643	0.2247
32	0.0231	22.0467	0.2605	0.221
33	0.0223	23.4977	0.2655	0.2259
34	0.0216	22.993	0.2655	0.2259
35	0.0205	22.5444	0.2569	0.2173
36	0.02	23.3201	0.2599	0.2203
37	0.0192	22.5841	0.2608	0.2213
38	0.0185	22.4019	0.2558	0.2163
39	0.0178	22.7009	0.2592	0.2197
40	0.0175	23.9322	0.2686	0.229
41	0.0162	23.507	0.2627	0.2231
42	0.0174	23.2009	0.2615	0.2219
43	0.0164	22.7266	0.2583	0.2187
44	0.0154	22.1285	0.2552	0.2156
45	0.0152	21.5607	0.2535	0.214
46	0.0148	22.2313	0.2544	0.2149
47	0.0141	22.9813	0.2631	0.2235
48	0.014	23.2523	0.2612	0.2217
49	0.0141	22.7874	0.2609	0.2213
50	0.0129	22.6285	0.2571	0.2175
51	0.0137	23.4229	0.2555	0.2159
52	0.0125	22.5164	0.2536	0.214
53	0.0127	21.8692	0.2468	0.2072
54	0.0124	22.6098	0.2561	0.2165
55	0.0117	21.75	0.2575	0.2179
56	0.0117	22.5888	0.2605	0.2209
57	0.0115	22.8224	0.2585	0.2189
58	0.0117	22.236	0.2607	0.2211
59	0.0112	22.6986	0.261	0.2214
60	0.0107	22.4883	0.2607	0.2212
61	0.0113	22.2547	0.2637	0.2242
--------------------------------------------------
MIN: 0.20720723821108175
MAX: 0.2802272338187018
--------------------------------------------------
