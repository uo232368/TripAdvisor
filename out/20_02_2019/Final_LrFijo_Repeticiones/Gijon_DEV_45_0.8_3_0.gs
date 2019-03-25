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
1	0.9299	26.6238	0.3009	0.2614
2	0.4565	28.8364	0.3118	0.2722
3	0.2834	26.3411	0.2938	0.2543
4	0.219	25.2593	0.2865	0.2469
5	0.1802	26.8715	0.2968	0.2573
6	0.1541	25.4509	0.288	0.2484
7	0.1341	23.2243	0.2759	0.2363
8	0.1213	25.2266	0.29	0.2504
9	0.109	24.9556	0.2844	0.2448
10	0.0999	25.9416	0.2812	0.2417
11	0.0933	25.5864	0.2775	0.2379
12	0.0871	25.7313	0.2848	0.2453
13	0.0813	25.1893	0.2821	0.2426
14	0.076	25.7173	0.2773	0.2377
15	0.0727	24.4907	0.2777	0.2381
16	0.0691	25.021	0.2801	0.2405
17	0.0663	25.8738	0.2913	0.2517
18	0.0647	25.6495	0.2802	0.2406
19	0.0603	25.1098	0.2834	0.2438
20	0.0578	25.1986	0.2842	0.2447
21	0.0554	24.7173	0.277	0.2375
22	0.054	25.9836	0.285	0.2454
23	0.0528	26.7383	0.2857	0.2461
24	0.05	26.1776	0.2927	0.2531
25	0.0483	25.8435	0.2827	0.2431
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
9	186	0.06626089036986867	0.13822786161793865
5	339	0.049065241601545226	0.15322797126835386
4	376	0.048995579266448055	0.152672704388138
2	428	0.039094627632103675	0.1373322084338339
1	428	0.039094627632103675	0.1373322084338339
----------------------------------------------------------------------------------------------------
26	0.0478	24.1963	0.284	0.2444
27	0.0463	24.3715	0.279	0.2394
28	0.045	25.5584	0.2811	0.2415
29	0.0434	24.3131	0.2828	0.2432
30	0.0429	26.1846	0.2884	0.2488
31	0.0413	25.8902	0.2867	0.2472
32	0.0407	26.3364	0.2854	0.2458
33	0.0402	25.535	0.2813	0.2417
34	0.0382	26.243	0.2825	0.2429
35	0.0377	26.3107	0.2882	0.2486
36	0.0379	25.6752	0.2806	0.241
37	0.0361	26.4042	0.2811	0.2415
38	0.0356	27.1051	0.2785	0.2389
39	0.0352	25.6799	0.2801	0.2405
40	0.0333	26.278	0.2821	0.2425
41	0.0335	26.1425	0.2868	0.2472
42	0.0326	26.1822	0.2815	0.242
43	0.0326	26.2664	0.2833	0.2437
44	0.0315	24.7967	0.2729	0.2334
45	0.0313	25.1215	0.2772	0.2377
46	0.0307	25.4766	0.2772	0.2376
47	0.0302	25.7103	0.2809	0.2414
48	0.0308	25.7453	0.2818	0.2422
49	0.0292	25.7243	0.2892	0.2496
50	0.028	25.1402	0.2866	0.247
51	0.0288	25.6121	0.2828	0.2433
52	0.0278	26.1355	0.2807	0.2411
--------------------------------------------------
MIN: 0.23337546863416758
MAX: 0.2721999357917874
--------------------------------------------------
