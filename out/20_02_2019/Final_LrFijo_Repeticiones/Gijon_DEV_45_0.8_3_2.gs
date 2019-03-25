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
db87a1b2fbde290c4112ae3a0838000f
--------------------------------------------------
[1mlearning_rate: [0m0.001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.9183	28.8715	0.3097	0.2701
2	0.4477	26.3645	0.2956	0.256
3	0.2734	25.0701	0.2799	0.2403
4	0.2106	24.7383	0.2725	0.2329
5	0.1732	23.236	0.2721	0.2325
6	0.1489	23.4813	0.2732	0.2336
7	0.1299	23.1682	0.271	0.2314
8	0.1164	24.4276	0.2727	0.2331
9	0.1046	23.8902	0.2729	0.2333
10	0.0981	25.5958	0.2749	0.2354
11	0.0893	26.1168	0.285	0.2454
12	0.0858	24.5234	0.2717	0.2321
13	0.0789	24.972	0.266	0.2264
14	0.0736	24.6682	0.2676	0.228
15	0.0692	23.8248	0.2685	0.229
16	0.0662	25.1192	0.2737	0.2341
17	0.0643	25.0234	0.2765	0.237
18	0.0592	25.3388	0.2808	0.2413
19	0.0576	25.5514	0.2762	0.2366
20	0.0555	24.4603	0.2772	0.2377
21	0.0536	24.243	0.2683	0.2287
22	0.052	25.9439	0.2789	0.2393
23	0.0493	25.1215	0.272	0.2324
24	0.048	25.0958	0.2645	0.2249
25	0.0467	25.1215	0.2712	0.2317
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
9	186	0.0900740666823575	0.16204103793042746
5	339	0.07004986734924831	0.17421259701605693
4	376	0.06310270023118576	0.16677982535287572
2	428	0.0512565746263037	0.14949415542803393
1	428	0.0512565746263037	0.14949415542803393
----------------------------------------------------------------------------------------------------
26	0.0445	24.6939	0.2718	0.2322
27	0.0423	24.9813	0.272	0.2325
28	0.042	24.0397	0.2642	0.2247
29	0.0414	24.2734	0.2662	0.2266
30	0.0401	24.4977	0.2713	0.2317
31	0.0388	24.9766	0.2709	0.2314
32	0.0377	23.6379	0.2596	0.22
33	0.0375	24.8014	0.2741	0.2345
34	0.035	25.4696	0.2728	0.2333
35	0.0356	24.1449	0.2672	0.2277
36	0.0349	24.4766	0.2684	0.2288
37	0.0336	23.5397	0.2696	0.23
38	0.0333	23.4486	0.2724	0.2328
39	0.0335	25.5	0.2718	0.2322
40	0.0322	24.3785	0.2677	0.2281
41	0.0305	24.5491	0.2695	0.2299
42	0.0312	24.271	0.2725	0.2329
43	0.0299	24.8505	0.27	0.2305
44	0.0289	24.4509	0.2712	0.2317
45	0.0289	24.6308	0.2748	0.2353
46	0.0293	24.0724	0.2715	0.232
47	0.0288	25.5584	0.2677	0.2281
48	0.0275	24.4579	0.2683	0.2288
49	0.0267	23.7547	0.2706	0.2311
50	0.0272	24.4299	0.2761	0.2365
--------------------------------------------------
MIN: 0.22000155450752956
MAX: 0.27009215689965677
--------------------------------------------------
