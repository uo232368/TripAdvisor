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
1	0.6022	20.1141	0.3033	0.2558
2	0.3275	19.9489	0.2962	0.2487
3	0.2654	20.3433	0.2986	0.2511
4	0.2289	20.0126	0.2976	0.2501
5	0.2041	19.9787	0.2945	0.247
6	0.1848	20.0713	0.2964	0.2489
7	0.1698	20.0978	0.2968	0.2493
8	0.1581	20.0644	0.2962	0.2487
9	0.1478	20.2149	0.2972	0.2498
10	0.1396	20.1315	0.2984	0.2509
11	0.1322	20.0176	0.2978	0.2504
12	0.1258	19.7168	0.2959	0.2484
13	0.1201	19.9583	0.2969	0.2494
14	0.1151	19.9269	0.2963	0.2488
15	0.1109	19.9283	0.2979	0.2504
16	0.1067	20.0124	0.2974	0.2499
17	0.1037	19.8891	0.2931	0.2456
18	0.1004	19.8753	0.296	0.2485
19	0.0973	20.305	0.299	0.2515
20	0.0947	20.0332	0.2983	0.2508
21	0.0922	19.942	0.296	0.2486
22	0.09	20.0472	0.2971	0.2496
23	0.088	19.9698	0.2959	0.2484
24	0.0858	20.1888	0.2954	0.2479
25	0.0843	20.4478	0.2987	0.2512
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
9	1891	0.43331805682859764	0.0672390929478205	0.19955336779931263	0.19864859699921453
5	3382	0.7749770852428964	0.05929593654393089	0.1786736906278127	0.2218602331231407
4	3733	0.8554078826764436	0.05830050225391774	0.17449571287919272	0.23040060481944122
2	4364	1.0	0.04853064806195358	0.16489215492875856	0.2501772288957252
1	4364	1.0	0.04853064806195358	0.16489215492875856	0.2501772288957252
----------------------------------------------------------------------------------------------------
26	0.0826	20.3217	0.2977	0.2502
27	0.0808	20.1904	0.2971	0.2496
28	0.0791	20.4929	0.2978	0.2503
29	0.0778	20.5974	0.3	0.2525
30	0.0762	20.2179	0.2956	0.2481
31	0.0753	20.4361	0.2991	0.2516
32	0.0741	20.3808	0.2991	0.2516
33	0.0732	20.6499	0.3001	0.2526
34	0.0721	20.5076	0.2995	0.252
35	0.0708	20.5188	0.2989	0.2514
36	0.0698	20.7124	0.3007	0.2532
37	0.0688	20.4732	0.2998	0.2523
38	0.0679	20.7949	0.3017	0.2542
39	0.0671	20.7385	0.3014	0.2539
40	0.066	20.1941	0.2984	0.251
41	0.0657	20.3721	0.2978	0.2503
42	0.0645	20.4283	0.3021	0.2546
43	0.0637	20.5621	0.3018	0.2543
44	0.0635	20.5958	0.3007	0.2532
45	0.0625	20.4393	0.3018	0.2543
46	0.0615	20.4051	0.3003	0.2528
47	0.0609	20.4292	0.3003	0.2528
48	0.0607	20.714	0.302	0.2545
49	0.0596	20.6936	0.3015	0.2541
50	0.0595	20.569	0.3013	0.2538
--------------------------------------------------
MIN: 0.24564427180079842
MAX: 0.2558470120870592
--------------------------------------------------
