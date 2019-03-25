Using TensorFlow backend.
[94mObteniendo datos...[0m
[93m[AVISO] 	Usuarios: 43628[0m
[93m[AVISO] 	Restaurantes: 6810[0m
[93m[AVISO] Cargando datos generados previamente...[0m
[94mCreando modelo...[0m


##################################################
 MODELV4
##################################################
 modelv4d2
##################################################
[93m[AVISO] Existen 1 combinaciones posibles[0m
--------------------------------------------------
c0567593df4f5f142b4f4e84597f517d
--------------------------------------------------
[1mlearning_rate: [0m0.0001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.87	40.3164	0.2938	0.2555
2	0.5723	37.5034	0.2795	0.2412
3	0.394	36.3591	0.2751	0.2367
4	0.2891	36.3446	0.2704	0.2321
5	0.2294	35.5207	0.2664	0.2281
6	0.1915	35.6226	0.2651	0.2268
7	0.1646	34.7583	0.2631	0.2247
8	0.1447	34.9954	0.2644	0.226
9	0.1293	34.814	0.2616	0.2233
10	0.1175	34.615	0.2613	0.223
11	0.1072	34.6501	0.2611	0.2228
12	0.099	34.3925	0.2597	0.2214
13	0.0926	34.0183	0.2605	0.2222
14	0.0865	34.0899	0.2584	0.2201
15	0.0811	34.1786	0.258	0.2197
16	0.0768	34.6212	0.2577	0.2193
17	0.0725	34.7039	0.2585	0.2202
18	0.0695	34.5342	0.258	0.2197
19	0.0662	34.4475	0.2588	0.2205
20	0.0632	34.3352	0.2586	0.2202
21	0.0608	34.5998	0.2567	0.2184
22	0.0586	35.028	0.259	0.2207
23	0.0562	34.7378	0.2586	0.2203
24	0.0544	34.8386	0.2582	0.2199
25	0.0527	34.5092	0.2568	0.2185
26	0.051	34.0403	0.2562	0.2179
27	0.0497	34.6738	0.2573	0.2189
28	0.0481	34.2043	0.255	0.2167
29	0.047	34.3899	0.2574	0.2191
30	0.0455	34.646	0.2568	0.2184
31	0.0443	35.1034	0.2556	0.2172
32	0.0433	35.1521	0.257	0.2187
33	0.0422	34.799	0.2562	0.2179
34	0.0412	35.0279	0.2568	0.2184
35	0.0405	34.9761	0.2569	0.2186
36	0.0395	34.6711	0.2558	0.2174
37	0.0387	34.5655	0.255	0.2167
38	0.038	34.7863	0.2543	0.216
39	0.0372	34.6535	0.2547	0.2163
40	0.0366	34.8561	0.2562	0.2179
41	0.0358	34.8756	0.2553	0.217
42	0.0354	34.6085	0.255	0.2167
43	0.0345	34.4108	0.2557	0.2174
44	0.0342	35.2838	0.2562	0.2179
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
N_FOTOS_TRAIN (>=)	N_ITEMS	%ITEMS	RND-MOD AC	CNT-MOD AC	MODELO
9	2568	0.438974358974359	0.08036861915964426	0.18535850009931035	0.16392927740698165
5	4590	0.7846153846153846	0.08046884104810699	0.18535213935538708	0.19221248414399286
4	5024	0.8588034188034188	0.08046884104810699	0.18535213935538708	0.19969287132498859
2	5850	1.0	0.08046884104810699	0.18535213935538708	0.21831749541774373
1	5850	1.0	0.08046884104810699	0.18535213935538708	0.21831749541774373
----------------------------------------------------------------------------------------------------
45	0.0334	35.2058	0.2566	0.2183
46	0.0329	34.7643	0.2554	0.217
47	0.0325	34.8123	0.2554	0.2171
48	0.0318	35.3985	0.2568	0.2184
49	0.0314	34.6979	0.2554	0.2171
50	0.0311	34.7966	0.2555	0.2172
51	0.0307	35.1308	0.2557	0.2174
52	0.0301	34.4309	0.2544	0.2161
53	0.0296	34.348	0.256	0.2177
54	0.0294	34.485	0.2556	0.2173
--------------------------------------------------
MIN: 0.21600874330632494
MAX: 0.25548583709147515
--------------------------------------------------
