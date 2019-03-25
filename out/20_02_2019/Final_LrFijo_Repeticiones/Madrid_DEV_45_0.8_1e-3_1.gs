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
1	0.5954	38.7142	0.2873	0.249
2	0.3137	38.2867	0.2794	0.2411
3	0.2549	38.0513	0.2808	0.2424
4	0.2215	38.6265	0.2816	0.2432
5	0.1981	38.0021	0.279	0.2407
6	0.1808	38.1764	0.2806	0.2423
7	0.1666	37.9879	0.2792	0.2408
8	0.1553	38.7126	0.2787	0.2404
9	0.1457	38.6477	0.2811	0.2427
10	0.1375	38.3241	0.2776	0.2393
11	0.1308	38.2171	0.28	0.2417
12	0.125	38.1788	0.2788	0.2405
13	0.1202	38.052	0.2784	0.24
14	0.1155	38.2981	0.2788	0.2405
15	0.111	38.6162	0.2783	0.24
16	0.1075	38.5407	0.2781	0.2398
17	0.1043	38.592	0.2799	0.2416
18	0.1013	37.9533	0.2795	0.2411
19	0.0981	37.593	0.2773	0.2389
20	0.0955	38.3255	0.2794	0.241
21	0.0933	39.2492	0.2808	0.2424
22	0.0914	37.8985	0.2767	0.2384
23	0.0896	38.7995	0.2787	0.2403
24	0.0873	37.9554	0.276	0.2377
25	0.0859	38.3528	0.2789	0.2405
26	0.0841	38.4284	0.2767	0.2384
27	0.0823	38.4113	0.2776	0.2393
28	0.0809	38.6236	0.2771	0.2388
29	0.0792	39.04	0.279	0.2407
30	0.0781	39.3316	0.28	0.2417
31	0.0771	39.3303	0.2779	0.2396
32	0.0758	38.9897	0.28	0.2417
33	0.0744	38.7674	0.2768	0.2385
34	0.0732	39.4089	0.2773	0.239
35	0.073	39.4019	0.2778	0.2394
36	0.0717	38.3942	0.2752	0.2369
37	0.0708	39.4456	0.2775	0.2392
38	0.0697	39.879	0.279	0.2407
39	0.069	39.2217	0.2767	0.2383
40	0.0678	39.3458	0.2753	0.2369
41	0.0671	39.1207	0.2757	0.2373
42	0.0663	39.2106	0.2753	0.237
43	0.0654	39.0892	0.2776	0.2392
44	0.0651	39.4402	0.2781	0.2398
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
9	2568	0.438974358974359	0.05887043504591373	0.1638603159855798	0.18610830190529373
5	4590	0.7846153846153846	0.05894267625043264	0.1638259745577127	0.21346279142656627
4	5024	0.8588034188034188	0.05894267625043264	0.1638259745577127	0.2219594197645499
2	5850	1.0	0.05894267625043264	0.1638259745577127	0.2398436602154181
1	5850	1.0	0.05894267625043264	0.1638259745577127	0.2398436602154181
----------------------------------------------------------------------------------------------------
45	0.0643	39.5178	0.2782	0.2398
46	0.0638	38.9176	0.2776	0.2393
47	0.0627	38.8318	0.2785	0.2402
48	0.0625	39.0918	0.2805	0.2422
49	0.0617	39.3682	0.2784	0.2401
50	0.0615	39.0383	0.2789	0.2406
--------------------------------------------------
MIN: 0.23691747835435129
MAX: 0.24899387066268933
--------------------------------------------------
