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
1	0.9484	26.1636	0.3047	0.2651
2	0.4761	28.8061	0.3075	0.2679
3	0.2712	25.2383	0.279	0.2394
4	0.206	24.7547	0.2766	0.237
5	0.1729	24.2243	0.2779	0.2384
6	0.1488	24.9416	0.2843	0.2447
7	0.1316	25.3411	0.2817	0.2421
8	0.1189	26.1332	0.2757	0.2361
9	0.1073	23.2079	0.2683	0.2287
10	0.0998	24.3248	0.2759	0.2363
11	0.0905	24.2734	0.2759	0.2364
12	0.087	24.0537	0.2785	0.2389
13	0.0822	22.9626	0.2799	0.2403
14	0.0762	21.764	0.2675	0.2279
15	0.0726	23.014	0.2789	0.2393
16	0.0696	23.3388	0.2705	0.231
17	0.0658	23.7126	0.2741	0.2346
18	0.0627	22.4369	0.2751	0.2355
19	0.0617	23.1168	0.2793	0.2397
20	0.0585	23.736	0.2781	0.2385
21	0.0558	23.0981	0.273	0.2334
22	0.0551	21.5234	0.2667	0.2272
23	0.0522	22.4019	0.2647	0.2251
24	0.051	22.9369	0.2765	0.237
25	0.0488	21.7757	0.2727	0.2331
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
9	186	0.07504923681109582	0.14701620805916576
5	339	0.06585364508225804	0.17001637474906667
4	376	0.058597387586763396	0.16227451270845333
2	428	0.045289569366122905	0.14352715016785314
1	428	0.045289569366122905	0.14352715016785314
----------------------------------------------------------------------------------------------------
26	0.0482	23.1379	0.2778	0.2382
27	0.0461	23.0444	0.2723	0.2327
28	0.0443	23.8621	0.277	0.2374
29	0.0436	23.3762	0.2803	0.2407
30	0.0424	21.2453	0.2653	0.2257
31	0.0414	22.5257	0.2692	0.2296
32	0.041	22.6168	0.2722	0.2327
33	0.0387	23.9276	0.2738	0.2342
34	0.0384	22.7266	0.2767	0.2371
35	0.0373	23.7757	0.2767	0.2371
36	0.0366	23.9603	0.2772	0.2376
37	0.036	23.7897	0.2783	0.2387
38	0.0358	24.6869	0.2802	0.2406
39	0.0354	23.3388	0.2774	0.2378
40	0.0341	23.6986	0.2772	0.2376
41	0.0345	23.7477	0.2748	0.2353
42	0.033	22.3037	0.274	0.2344
43	0.0318	23.1799	0.2819	0.2423
44	0.0316	24.4533	0.2825	0.2429
45	0.0313	25.5771	0.2828	0.2432
46	0.0311	23.9603	0.2833	0.2437
47	0.0298	24.0537	0.2866	0.247
48	0.0301	24.5234	0.2792	0.2396
49	0.0302	22.9299	0.2747	0.2351
50	0.0287	24.3201	0.2796	0.24
--------------------------------------------------
MIN: 0.22509983974876943
MAX: 0.2678979717637013
--------------------------------------------------
