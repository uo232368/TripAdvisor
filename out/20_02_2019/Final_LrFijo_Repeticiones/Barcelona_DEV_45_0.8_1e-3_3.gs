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
1	0.6313	21.9413	0.3178	0.2704
2	0.3418	20.8197	0.3023	0.2548
3	0.2718	20.8362	0.299	0.2515
4	0.2309	20.6343	0.301	0.2535
5	0.2026	20.8639	0.3029	0.2554
6	0.1823	20.5037	0.302	0.2545
7	0.1664	20.6907	0.3032	0.2557
8	0.1542	20.6659	0.2993	0.2518
9	0.1441	20.8857	0.3014	0.2539
10	0.1357	20.7216	0.3016	0.2541
11	0.1283	20.8071	0.2986	0.2511
12	0.1224	20.6434	0.3016	0.2541
13	0.117	20.684	0.301	0.2535
14	0.1119	20.56	0.2978	0.2503
15	0.1079	20.7388	0.3009	0.2535
16	0.1041	20.5646	0.3015	0.2541
17	0.1004	20.8662	0.3005	0.253
18	0.0977	20.9805	0.3023	0.2549
19	0.0944	21.0575	0.3065	0.259
20	0.0926	21.0268	0.3054	0.2579
21	0.0899	21.1874	0.306	0.2585
22	0.0873	20.9569	0.3038	0.2563
23	0.0855	21.2049	0.3058	0.2583
24	0.0833	21.0222	0.3039	0.2564
25	0.0816	20.8838	0.3019	0.2544
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
9	1891	0.43331805682859764	0.06458276612323195	0.19689704097472405	0.20130492382380305
5	3382	0.7749770852428964	0.058020253912222625	0.17739800799610442	0.223135915754849
4	3733	0.8554078826764436	0.05618153311448851	0.17237674373976347	0.2325195739588704
2	4364	1.0	0.04509063977224059	0.16145214663904559	0.25361723718543816
1	4364	1.0	0.04509063977224059	0.16145214663904559	0.25361723718543816
----------------------------------------------------------------------------------------------------
26	0.0799	20.8412	0.3011	0.2536
27	0.0781	20.742	0.2993	0.2518
28	0.0772	20.7915	0.3013	0.2538
29	0.0757	21.1939	0.3045	0.257
30	0.0743	20.9677	0.3021	0.2546
31	0.0728	21.1971	0.3059	0.2584
32	0.0717	21.1384	0.3023	0.2548
33	0.0707	21.0399	0.3029	0.2554
34	0.0698	20.9947	0.3021	0.2546
35	0.0688	21.0825	0.3025	0.255
36	0.0679	20.8994	0.3028	0.2553
37	0.0671	21.2005	0.3041	0.2566
38	0.0661	21.1329	0.3028	0.2553
39	0.0649	21.0978	0.3038	0.2563
40	0.0644	21.4166	0.3038	0.2564
41	0.0636	21.3765	0.305	0.2575
42	0.0629	21.1517	0.3045	0.257
43	0.0619	21.225	0.306	0.2585
44	0.0611	21.1235	0.304	0.2565
45	0.0606	21.2294	0.3051	0.2576
46	0.0599	21.2486	0.3063	0.2588
47	0.059	21.2658	0.3051	0.2576
48	0.0588	21.4157	0.3069	0.2594
49	0.0582	21.189	0.3054	0.2579
50	0.0576	21.4879	0.3046	0.2571
--------------------------------------------------
MIN: 0.2503459643008773
MAX: 0.27035248664314426
--------------------------------------------------
