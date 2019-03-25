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
769431f98abdc10591cfe7b5de2a53a3
--------------------------------------------------
[1mlearning_rate: [0m0.001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.576	39.2017	0.2873	0.249
2	0.3153	39.3889	0.2864	0.2481
3	0.2583	38.441	0.2866	0.2483
4	0.2245	38.8865	0.2847	0.2464
5	0.2013	38.8051	0.2838	0.2455
6	0.1836	39.6009	0.2832	0.2449
7	0.1697	38.6913	0.2826	0.2442
8	0.1586	39.0583	0.2838	0.2455
9	0.1495	38.7504	0.2821	0.2437
10	0.1416	39.6933	0.2846	0.2463
11	0.1349	38.8998	0.2835	0.2452
12	0.1294	38.9221	0.287	0.2486
13	0.1242	38.7964	0.2851	0.2467
14	0.1201	39.4097	0.2865	0.2482
15	0.1156	39.4434	0.2851	0.2468
16	0.1118	39.5904	0.286	0.2476
17	0.1086	39.7434	0.2859	0.2475
18	0.1058	40.2316	0.2852	0.2468
19	0.1031	39.3496	0.2835	0.2452
20	0.1007	39.7434	0.2862	0.2479
21	0.0983	39.2391	0.2849	0.2466
22	0.0959	39.6643	0.2852	0.2469
23	0.0938	39.7619	0.2857	0.2474
24	0.0921	40.0728	0.289	0.2506
25	0.0901	40.5638	0.2898	0.2515
26	0.0885	40.1463	0.2877	0.2494
27	0.0874	40.4978	0.2863	0.2479
28	0.086	40.8839	0.2893	0.251
29	0.0845	40.2827	0.2856	0.2473
30	0.083	39.9802	0.2879	0.2495
31	0.0819	40.4827	0.2887	0.2504
32	0.0807	40.735	0.2889	0.2506
33	0.0794	40.0762	0.2888	0.2505
34	0.0784	40.6238	0.2901	0.2518
35	0.0777	40.1803	0.2904	0.2521
36	0.0766	40.8947	0.2931	0.2548
37	0.0754	40.5821	0.293	0.2547
38	0.0745	41.4426	0.291	0.2527
39	0.0738	41.9032	0.2936	0.2553
40	0.073	40.8178	0.2921	0.2538
41	0.072	41.6812	0.293	0.2547
42	0.0717	40.468	0.292	0.2536
43	0.0704	41.0757	0.292	0.2537
44	0.0701	41.3595	0.2906	0.2523
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
9	2568	0.438974358974359	0.04679616645216575	0.15178604739183182	0.195352913429491
5	4590	0.7846153846153846	0.04684514716445312	0.15172844547173317	0.22594023272983887
4	5024	0.8588034188034188	0.04684514716445312	0.15172844547173317	0.23262308388860484
2	5850	1.0	0.04684514716445312	0.15172844547173317	0.2519411893013976
1	5850	1.0	0.04684514716445312	0.15172844547173317	0.2519411893013976
----------------------------------------------------------------------------------------------------
45	0.0693	40.3101	0.2903	0.2519
46	0.0686	41.3074	0.2922	0.2539
47	0.0677	40.5692	0.2914	0.253
48	0.0672	40.6104	0.2918	0.2534
49	0.0665	40.7477	0.2907	0.2524
50	0.0662	40.8593	0.2917	0.2534
--------------------------------------------------
MIN: 0.24373652507173874
MAX: 0.25525222745988263
--------------------------------------------------
