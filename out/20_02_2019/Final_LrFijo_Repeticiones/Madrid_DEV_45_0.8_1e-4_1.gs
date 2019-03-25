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
c0567593df4f5f142b4f4e84597f517d
--------------------------------------------------
[1mlearning_rate: [0m0.0001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.9033	40.6121	0.2968	0.2585
2	0.5883	38.2708	0.2802	0.2419
3	0.4013	36.4456	0.2747	0.2364
4	0.2964	36.5807	0.2713	0.233
5	0.2369	36.5544	0.2723	0.234
6	0.1968	35.8289	0.2685	0.2302
7	0.1689	36.6788	0.2681	0.2298
8	0.1481	36.7687	0.2676	0.2292
9	0.1315	36.3357	0.2652	0.2269
10	0.1192	36.0103	0.2629	0.2245
11	0.1088	35.8106	0.2618	0.2234
12	0.1003	35.7915	0.2624	0.224
13	0.0929	35.9884	0.2621	0.2238
14	0.0867	35.9315	0.2615	0.2232
15	0.0818	36.4472	0.2621	0.2238
16	0.0774	36.1586	0.2606	0.2222
17	0.073	36.8938	0.2619	0.2236
18	0.0697	36.4156	0.26	0.2217
19	0.0664	36.8937	0.2607	0.2224
20	0.0633	36.5256	0.261	0.2227
21	0.0607	35.5985	0.2577	0.2193
22	0.0586	35.4197	0.2576	0.2192
23	0.0566	35.8521	0.258	0.2197
24	0.0543	36.3976	0.2583	0.22
25	0.0529	36.6605	0.2587	0.2203
26	0.051	36.3321	0.2577	0.2193
27	0.0494	36.0499	0.258	0.2196
28	0.0478	35.8044	0.258	0.2197
29	0.0466	36.1648	0.2563	0.218
30	0.0453	35.9405	0.258	0.2197
31	0.0445	35.7221	0.2567	0.2184
32	0.0433	35.9041	0.2575	0.2192
33	0.0421	35.7407	0.2581	0.2197
34	0.0411	36.2634	0.2578	0.2194
35	0.04	35.9636	0.257	0.2187
36	0.0394	36.2776	0.2588	0.2204
37	0.0382	36.6161	0.2571	0.2188
38	0.0377	36.2332	0.2577	0.2194
39	0.0372	36.5937	0.2567	0.2184
40	0.0363	35.9569	0.2555	0.2171
41	0.0357	35.9051	0.2562	0.2179
42	0.0348	36.0056	0.2563	0.218
43	0.0344	35.8723	0.2562	0.2179
44	0.0336	35.9781	0.2551	0.2168
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
9	2568	0.438974358974359	0.08325565369111959	0.18824553463078567	0.15959207885856663
5	4590	0.7846153846153846	0.08331739536475262	0.18820069367203265	0.1890617520163584
4	5024	0.8588034188034188	0.08331739536475262	0.18820069367203265	0.19595259931789152
2	5850	1.0	0.08331739536475262	0.18820069367203265	0.21546894110109807
1	5850	1.0	0.08331739536475262	0.18820069367203265	0.21546894110109807
----------------------------------------------------------------------------------------------------
45	0.0333	36.7672	0.2538	0.2155
46	0.0328	36.9026	0.2561	0.2178
47	0.0323	36.2532	0.2565	0.2182
48	0.0317	35.6108	0.2545	0.2162
49	0.0312	35.4282	0.2543	0.216
50	0.0307	36.6901	0.2537	0.2154
51	0.0305	36.4015	0.2534	0.2151
52	0.0299	35.8342	0.254	0.2157
53	0.0294	36.021	0.2547	0.2164
54	0.029	36.4448	0.2569	0.2186
55	0.0288	35.6744	0.2548	0.2165
56	0.0284	35.6056	0.2551	0.2168
57	0.028	36.1561	0.2541	0.2158
58	0.0276	36.3133	0.2546	0.2163
59	0.0274	36.8089	0.2541	0.2158
60	0.0271	36.6253	0.2555	0.2172
61	0.0266	35.9991	0.2538	0.2155
62	0.0264	36.1547	0.2548	0.2165
63	0.0258	35.4636	0.2553	0.217
64	0.0258	35.2085	0.2545	0.2161
65	0.0253	35.3403	0.2543	0.2159
66	0.0251	36.5325	0.2545	0.2162
67	0.0249	36.4031	0.255	0.2167
--------------------------------------------------
MIN: 0.21511352665853015
MAX: 0.258479530693605
--------------------------------------------------
