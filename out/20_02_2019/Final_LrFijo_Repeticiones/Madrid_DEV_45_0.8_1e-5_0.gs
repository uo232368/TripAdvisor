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
15a666425f322f704c77c6433bc64f17
--------------------------------------------------
[1mlearning_rate: [0m1e-05
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	1.0073	49.2084	0.3559	0.3176
2	0.9904	47.5826	0.3495	0.3112
3	0.9762	45.5863	0.3388	0.3005
4	0.9173	43.6622	0.3199	0.2816
5	0.8109	40.9639	0.3055	0.2672
6	0.7273	39.7981	0.2953	0.257
7	0.662	38.7246	0.2877	0.2493
8	0.6101	37.5993	0.2812	0.2428
9	0.5684	36.8568	0.278	0.2397
10	0.5323	36.4629	0.2758	0.2375
11	0.5006	36.3138	0.274	0.2356
12	0.4722	36.3923	0.273	0.2347
13	0.4463	36.1718	0.2714	0.2331
14	0.4225	36.4318	0.2708	0.2324
15	0.3996	36.6964	0.2705	0.2321
16	0.3781	36.5186	0.2693	0.231
17	0.359	36.6674	0.2687	0.2304
18	0.3407	36.5713	0.2684	0.2301
19	0.3242	36.412	0.2678	0.2294
20	0.3086	36.5521	0.2665	0.2282
21	0.2935	36.425	0.266	0.2277
22	0.2799	36.2333	0.2649	0.2266
23	0.2675	36.4788	0.2646	0.2263
24	0.2553	36.7265	0.2649	0.2266
25	0.2447	36.6456	0.2643	0.2259
26	0.2343	36.6812	0.2632	0.2248
27	0.2253	36.5525	0.2627	0.2244
28	0.2165	36.5262	0.2631	0.2248
29	0.2084	36.2361	0.262	0.2237
30	0.2006	36.465	0.2621	0.2238
31	0.193	36.3368	0.2608	0.2225
32	0.1865	36.2615	0.2612	0.2229
33	0.1801	36.0152	0.2605	0.2221
34	0.1738	36.0108	0.2607	0.2224
35	0.1684	36.1632	0.2605	0.2222
36	0.1624	36.2819	0.2599	0.2215
37	0.157	36.3137	0.2606	0.2223
38	0.1527	36.172	0.2604	0.2221
39	0.1482	36.4239	0.2594	0.2211
40	0.1435	36.2865	0.2597	0.2214
41	0.1393	36.0106	0.2588	0.2205
42	0.1356	36.0694	0.2583	0.22
43	0.1321	36.0957	0.2586	0.2202
44	0.1286	35.6561	0.258	0.2197
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
9	2568	0.438974358974359	0.07862695432062634	0.1836168352602924	0.16579572859418865
5	4590	0.7846153846153846	0.07873380504701662	0.18361710335429665	0.19335588285209507
4	5024	0.8588034188034188	0.07873380504701662	0.18361710335429665	0.20047164566127054
2	5850	1.0	0.07873380504701662	0.18361710335429665	0.22005253141883413
1	5850	1.0	0.07873380504701662	0.18361710335429665	0.22005253141883413
----------------------------------------------------------------------------------------------------
45	0.125	36.0383	0.2584	0.2201
46	0.1219	35.8677	0.2576	0.2193
47	0.119	36.265	0.2579	0.2195
48	0.1157	35.9865	0.2584	0.2201
49	0.113	36.0091	0.2579	0.2196
50	0.1103	35.9395	0.2575	0.2192
51	0.1078	35.8005	0.258	0.2197
52	0.1054	35.9986	0.2582	0.2199
53	0.1027	35.5356	0.2577	0.2194
54	0.1006	35.9159	0.2572	0.2189
55	0.0982	35.6173	0.2573	0.2189
56	0.0962	35.6779	0.2571	0.2188
57	0.0943	35.981	0.2569	0.2186
58	0.0925	35.8781	0.2577	0.2194
59	0.0901	35.8709	0.2581	0.2198
60	0.0884	35.7497	0.257	0.2187
61	0.0868	36.0036	0.258	0.2196
62	0.0852	35.9439	0.257	0.2187
63	0.0833	35.5537	0.2571	0.2188
64	0.0816	35.6012	0.2569	0.2186
65	0.0802	35.7942	0.2567	0.2184
66	0.0791	35.8458	0.256	0.2177
67	0.0773	35.7431	0.2561	0.2178
68	0.0761	35.7979	0.2565	0.2182
69	0.0742	35.4856	0.2565	0.2182
70	0.0734	35.7101	0.2567	0.2184
71	0.0718	36.0998	0.2571	0.2188
72	0.071	35.6533	0.2558	0.2175
73	0.0697	35.6959	0.2566	0.2183
74	0.0686	35.7179	0.256	0.2177
75	0.0674	35.4797	0.2559	0.2176
76	0.0663	35.4872	0.256	0.2177
77	0.0654	35.5752	0.2558	0.2174
78	0.0644	35.5761	0.2569	0.2185
79	0.0633	35.6084	0.2555	0.2172
80	0.0621	35.4176	0.2554	0.217
81	0.0614	35.5239	0.2554	0.2171
82	0.0602	35.5364	0.2566	0.2183
83	0.0594	35.3745	0.2554	0.217
84	0.0589	35.3338	0.2562	0.2179
85	0.0579	35.7017	0.2562	0.2179
86	0.0573	35.6296	0.2561	0.2177
87	0.0566	35.4342	0.2555	0.2172
88	0.0555	35.5147	0.2559	0.2175
89	0.0546	35.319	0.2561	0.2178
90	0.0541	35.4783	0.2562	0.2179
91	0.0533	35.0928	0.2551	0.2168
92	0.0525	35.2554	0.2555	0.2172
93	0.0519	35.4352	0.2559	0.2175
94	0.0508	35.4692	0.2547	0.2163
95	0.0502	35.6809	0.2556	0.2173
96	0.0501	35.5725	0.2558	0.2174
97	0.0493	35.7467	0.2554	0.2171
98	0.0486	35.5171	0.2553	0.2169
99	0.0479	35.4899	0.2556	0.2173
100	0.0475	35.6725	0.2557	0.2174
101	0.0468	35.4154	0.256	0.2177
102	0.0461	35.6771	0.2556	0.2173
103	0.0457	35.4376	0.256	0.2177
104	0.0453	35.3304	0.2557	0.2174
105	0.0444	35.3906	0.2562	0.2179
--------------------------------------------------
MIN: 0.21633591245220124
MAX: 0.3176004960745267
--------------------------------------------------
