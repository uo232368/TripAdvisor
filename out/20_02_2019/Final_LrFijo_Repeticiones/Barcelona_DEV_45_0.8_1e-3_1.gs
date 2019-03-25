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
1	0.6144	20.5637	0.3001	0.2526
2	0.3165	19.7981	0.2926	0.2451
3	0.2509	19.6425	0.29	0.2425
4	0.214	19.9338	0.2905	0.243
5	0.1877	20.0846	0.2919	0.2444
6	0.1688	19.8025	0.2902	0.2427
7	0.1539	19.7241	0.2917	0.2442
8	0.1416	19.8444	0.2909	0.2434
9	0.1323	19.712	0.2923	0.2448
10	0.124	19.8123	0.2885	0.241
11	0.1176	19.9299	0.2917	0.2442
12	0.1111	19.6201	0.2887	0.2412
13	0.1066	19.8304	0.2904	0.2429
14	0.1023	20.0312	0.2921	0.2446
15	0.0982	19.8016	0.2918	0.2443
16	0.0948	20.0736	0.2917	0.2442
17	0.0916	19.9973	0.2921	0.2446
18	0.0886	19.5756	0.2914	0.244
19	0.0859	19.8417	0.2906	0.2431
20	0.0839	19.5752	0.2885	0.241
21	0.0814	19.5158	0.2888	0.2413
22	0.0791	19.7243	0.2909	0.2434
23	0.0778	19.7454	0.2898	0.2423
24	0.076	19.5616	0.2903	0.2428
25	0.0744	19.9127	0.2929	0.2454
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
9	1891	0.43331805682859764	0.0705792656581944	0.20289354050968653	0.1953084242888406
5	3382	0.7749770852428964	0.06546163474529748	0.18483938882917927	0.21569453492177415
4	3733	0.8554078826764436	0.06380259530403096	0.17999780592930592	0.22489851176932799
2	4364	1.0	0.054299637881548304	0.1706611447483533	0.24440823907613046
1	4364	1.0	0.054299637881548304	0.1706611447483533	0.24440823907613046
----------------------------------------------------------------------------------------------------
26	0.0729	19.8607	0.2919	0.2444
27	0.0713	19.6354	0.2895	0.242
28	0.0702	19.5069	0.2897	0.2422
29	0.0684	19.8112	0.2907	0.2432
30	0.0674	19.8261	0.292	0.2445
31	0.0659	19.5472	0.2889	0.2414
32	0.0651	19.646	0.2901	0.2426
33	0.0642	19.7663	0.2933	0.2458
34	0.0632	19.7793	0.2913	0.2439
35	0.062	19.7333	0.291	0.2435
36	0.0614	19.929	0.2926	0.2451
37	0.0604	19.4464	0.2906	0.2431
38	0.0597	19.5121	0.2884	0.2409
39	0.0592	19.8708	0.2935	0.246
40	0.0581	19.9232	0.2899	0.2424
41	0.0575	19.8669	0.2877	0.2402
42	0.0567	19.956	0.2895	0.242
43	0.0563	20.0919	0.2922	0.2447
44	0.0555	19.9663	0.2919	0.2444
45	0.0547	19.6634	0.2904	0.2429
46	0.0541	19.78	0.2915	0.244
47	0.0537	19.9954	0.2931	0.2456
48	0.0534	19.7764	0.2905	0.243
49	0.0526	19.9464	0.2938	0.2463
50	0.0521	19.9432	0.292	0.2445
--------------------------------------------------
MIN: 0.24017476844905397
MAX: 0.25260922533258945
--------------------------------------------------
