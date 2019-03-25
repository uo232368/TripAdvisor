Using TensorFlow backend.
[94mObteniendo datos...[0m
[93m[AVISO] 	Usuarios: 33537[0m
[93m[AVISO] 	Restaurantes: 5881[0m
[93m[AVISO] Cargando datos generados previamente...[0m
[94mCreando modelo...[0m


##################################################
 MODELV4
##################################################
 modelv4d2
##################################################
[93m[AVISO] Existen 1 combinaciones posibles[0m
--------------------------------------------------
9d5322f45240b9639fa6217d77f0ba38
--------------------------------------------------
[1mlearning_rate: [0m0.001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.6311	20.4083	0.2982	0.2507
2	0.3309	20.6884	0.3009	0.2534
3	0.27	20.1217	0.2974	0.2499
4	0.2354	20.0348	0.2979	0.2504
5	0.2104	20.2305	0.2995	0.252
6	0.1917	20.1499	0.296	0.2485
7	0.176	20.6467	0.3016	0.2542
8	0.1643	20.5623	0.2991	0.2517
9	0.1543	20.5694	0.3005	0.253
10	0.1453	20.5082	0.3007	0.2532
11	0.1384	20.7447	0.303	0.2555
12	0.1323	21.0257	0.3012	0.2537
13	0.1261	20.662	0.3018	0.2543
14	0.1214	20.7617	0.304	0.2565
15	0.117	20.874	0.3035	0.256
16	0.1131	21.1737	0.3068	0.2594
17	0.1093	21.0364	0.3039	0.2564
18	0.1059	21.0025	0.3022	0.2547
19	0.1034	21.1865	0.3052	0.2577
20	0.1002	21.0658	0.3058	0.2583
21	0.0978	20.945	0.3047	0.2572
22	0.0952	21.1233	0.3043	0.2569
23	0.0933	21.2697	0.3051	0.2576
24	0.0915	21.1753	0.3059	0.2584
25	0.0893	20.8673	0.3033	0.2558
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
9	1891	0.43331805682859764	0.05932762374755054	0.19164189859904265	0.20656006619948447
5	3382	0.7749770852428964	0.0561888025927069	0.1755665566765887	0.2249673670743647
4	3733	0.8554078826764436	0.05526324492396943	0.1714584555492444	0.2334378621493895
2	4364	1.0	0.04279383660468545	0.15915534347149046	0.2559140403529933
1	4364	1.0	0.04279383660468545	0.15915534347149046	0.2559140403529933
----------------------------------------------------------------------------------------------------
26	0.0874	21.1299	0.3034	0.2559
27	0.0856	21.4553	0.3076	0.2601
28	0.0845	21.0901	0.3031	0.2556
29	0.0827	21.0919	0.3038	0.2563
30	0.0814	20.9196	0.3029	0.2554
31	0.0797	21.0066	0.3029	0.2554
32	0.0786	21.0456	0.3034	0.2559
33	0.0775	20.9883	0.3028	0.2553
34	0.0765	21.2298	0.3046	0.2571
35	0.075	20.9452	0.3041	0.2566
36	0.0745	20.8137	0.3001	0.2526
37	0.0731	21.0667	0.3019	0.2544
38	0.0722	20.8577	0.3002	0.2527
39	0.0713	20.8499	0.2998	0.2523
40	0.0705	20.8025	0.3006	0.2531
41	0.0695	20.9918	0.3031	0.2556
42	0.0691	20.6386	0.299	0.2515
43	0.068	20.8132	0.303	0.2555
44	0.0669	20.7656	0.3028	0.2554
45	0.0664	20.8703	0.3018	0.2543
46	0.0655	21.0115	0.3019	0.2544
47	0.0649	20.607	0.3025	0.255
48	0.0644	20.8611	0.3035	0.256
49	0.0638	20.9773	0.3031	0.2556
50	0.0634	20.9535	0.304	0.2566
--------------------------------------------------
MIN: 0.24848953758708564
MAX: 0.2600914288304329
--------------------------------------------------
