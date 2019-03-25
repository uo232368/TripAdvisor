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
1	1.0029	28.4883	0.3247	0.2851
2	0.9428	26.4182	0.3017	0.2622
3	0.7295	26.5117	0.293	0.2534
4	0.5523	24.9393	0.281	0.2414
5	0.3977	24.9229	0.2791	0.2395
6	0.2927	25.0935	0.2813	0.2417
7	0.2269	24.5701	0.2785	0.2389
8	0.1845	23.1916	0.2715	0.2319
9	0.153	23.8435	0.2686	0.229
10	0.1297	23.3738	0.2693	0.2298
11	0.1131	23.6893	0.2676	0.228
12	0.0978	24.2033	0.2727	0.2331
13	0.0885	23.5584	0.2658	0.2262
14	0.0797	23.2173	0.2666	0.227
15	0.071	22.8598	0.2587	0.2191
16	0.0648	22.4019	0.2589	0.2193
17	0.0586	22.4743	0.2653	0.2257
18	0.0537	22.8481	0.2677	0.2281
19	0.0497	23.4836	0.2633	0.2237
20	0.0454	23.1636	0.2624	0.2228
21	0.0428	22.6285	0.2643	0.2247
22	0.04	23.1192	0.2625	0.2229
23	0.038	22.778	0.262	0.2225
24	0.036	22.9276	0.2592	0.2196
25	0.0329	22.1425	0.2615	0.2219
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
9	186	0.10578624406924673	0.1777532153173167
5	339	0.07677595560498607	0.18093868527179469
4	376	0.07433795172029774	0.1780150768419877
2	428	0.06549598861022489	0.16373356941195516
1	428	0.06549598861022489	0.16373356941195516
----------------------------------------------------------------------------------------------------
26	0.0314	22.7477	0.2576	0.218
27	0.0301	23.4813	0.2619	0.2223
28	0.029	21.6893	0.2561	0.2165
29	0.0272	23.1963	0.2608	0.2213
30	0.0266	22.5818	0.2663	0.2267
31	0.0249	22.7827	0.2592	0.2197
32	0.0238	22.5257	0.2623	0.2227
33	0.0232	22.8341	0.2611	0.2215
34	0.0221	22.3411	0.2624	0.2228
35	0.0207	22.9416	0.2609	0.2214
36	0.021	22.6822	0.2595	0.22
37	0.0204	22.4463	0.2588	0.2193
38	0.0195	22.5958	0.261	0.2214
39	0.0184	22.9136	0.2599	0.2204
40	0.0177	22.1636	0.2591	0.2196
41	0.0183	22.1028	0.2575	0.2179
42	0.0169	22.0935	0.2577	0.2182
43	0.0166	22.6542	0.2559	0.2163
44	0.016	22.1589	0.2555	0.2159
45	0.0156	22.2477	0.2569	0.2173
46	0.0153	22.3154	0.2535	0.2139
47	0.0148	22.2033	0.2629	0.2234
48	0.0143	22.3598	0.2584	0.2188
49	0.0145	22.8575	0.2559	0.2164
50	0.0137	22.4977	0.2618	0.2222
51	0.0136	23.4369	0.2656	0.226
52	0.0135	22.3972	0.2614	0.2218
53	0.0129	21.8621	0.2582	0.2186
54	0.0125	22.4953	0.26	0.2204
--------------------------------------------------
MIN: 0.21390300898966638
MAX: 0.28514404919451364
--------------------------------------------------
