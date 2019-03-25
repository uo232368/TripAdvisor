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
1	0.918	26.6215	0.2989	0.2593
2	0.4134	25.986	0.2786	0.239
3	0.2543	27.222	0.2853	0.2457
4	0.1987	26.785	0.2849	0.2454
5	0.1674	26.2056	0.279	0.2394
6	0.1441	25.986	0.2763	0.2367
7	0.1277	26.472	0.2813	0.2418
8	0.1168	25.7243	0.2775	0.2379
9	0.105	25.3551	0.2766	0.2371
10	0.0968	25.3178	0.2745	0.2349
11	0.091	24.6215	0.2765	0.2369
12	0.0849	24.8575	0.2737	0.2341
13	0.0798	24.507	0.2714	0.2318
14	0.0764	27.0397	0.2824	0.2429
15	0.0714	26.5631	0.2786	0.239
16	0.0676	25.9416	0.2751	0.2356
17	0.0639	25.7921	0.2796	0.2401
18	0.062	24.1332	0.2678	0.2282
19	0.0589	25.7944	0.2852	0.2456
20	0.0576	24.8879	0.2732	0.2336
21	0.0547	25.2033	0.2767	0.2371
22	0.0536	25.3364	0.2794	0.2399
23	0.0512	25.5467	0.2818	0.2422
24	0.0499	26.8364	0.2868	0.2472
25	0.0484	25.3949	0.2803	0.2407
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
9	186	0.08415599777697866	0.1561229690250486
5	339	0.06063914123765172	0.16480187090446038
4	376	0.060153783991561904	0.16383090911325182
2	428	0.046082766072719654	0.1443203468744499
1	428	0.046082766072719654	0.1443203468744499
----------------------------------------------------------------------------------------------------
26	0.0468	25.0164	0.277	0.2374
27	0.046	23.9112	0.2769	0.2373
28	0.0435	25.6098	0.2774	0.2378
29	0.0437	25.7477	0.279	0.2394
30	0.0418	25.5771	0.2789	0.2394
31	0.0405	25.7804	0.2855	0.2459
32	0.0394	25.215	0.2882	0.2486
33	0.0392	25.2009	0.2825	0.2429
34	0.0383	24.9673	0.2775	0.238
35	0.0365	24.3762	0.2818	0.2422
36	0.0371	25.722	0.2833	0.2437
37	0.0354	24.8458	0.2854	0.2459
38	0.0351	26.0514	0.2948	0.2552
39	0.0342	25.5397	0.2921	0.2525
40	0.0342	25.2266	0.2833	0.2437
41	0.0324	24.4813	0.285	0.2455
42	0.0326	24.5397	0.2861	0.2466
43	0.0327	24.8528	0.2854	0.2458
44	0.0319	24.3505	0.283	0.2434
45	0.031	24.7126	0.2734	0.2338
46	0.0303	25.257	0.2828	0.2432
47	0.0291	24.9416	0.2805	0.2409
48	0.0292	24.3668	0.2827	0.2432
49	0.0293	24.6682	0.277	0.2374
50	0.0291	25.7593	0.2804	0.2408
51	0.0281	24.8014	0.2748	0.2352
52	0.0273	23.8014	0.2636	0.224
53	0.0268	23.9089	0.2716	0.2321
54	0.0272	25.6986	0.2795	0.2399
55	0.027	24.3107	0.2779	0.2383
56	0.0262	24.8879	0.2814	0.2418
57	0.0255	24.9673	0.2764	0.2369
58	0.0256	24.3107	0.2732	0.2337
59	0.0251	24.771	0.2777	0.2382
60	0.0254	24.8294	0.2749	0.2354
61	0.0251	25.3154	0.2725	0.2329
62	0.0239	23.8808	0.2709	0.2313
63	0.0239	23.7967	0.2672	0.2277
64	0.0237	24.2921	0.2712	0.2317
65	0.0239	24.3762	0.2744	0.2349
66	0.0243	23.8271	0.2681	0.2285
67	0.0229	25.243	0.2737	0.2341
68	0.0226	24.8271	0.2819	0.2423
69	0.0222	22.9673	0.2722	0.2326
70	0.0223	25.3949	0.2743	0.2347
71	0.0232	23.7593	0.2748	0.2353
72	0.0214	22.5841	0.2687	0.2292
73	0.0216	23.0818	0.2685	0.2289
74	0.0214	23.25	0.2779	0.2384
75	0.021	24.1659	0.2773	0.2377
76	0.0218	23.6121	0.2654	0.2258
77	0.0206	24.0257	0.268	0.2284
78	0.0199	23.9509	0.271	0.2314
79	0.0211	23.9322	0.2778	0.2382
--------------------------------------------------
MIN: 0.22401740194231737
MAX: 0.25934872418405913
--------------------------------------------------
