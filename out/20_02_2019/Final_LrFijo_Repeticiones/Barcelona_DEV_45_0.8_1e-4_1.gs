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
26e6244bc5b369fa7515f3aad40a2505
--------------------------------------------------
[1mlearning_rate: [0m0.0001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.9354	22.1008	0.3174	0.2699
2	0.6406	20.0548	0.2968	0.2494
3	0.4543	19.1068	0.2867	0.2392
4	0.3309	19.0775	0.2829	0.2354
5	0.2595	18.8126	0.2788	0.2313
6	0.2127	18.7993	0.2774	0.2299
7	0.1808	18.8728	0.2762	0.2288
8	0.1569	18.8203	0.2755	0.228
9	0.1389	18.5779	0.2718	0.2243
10	0.1247	18.5642	0.2713	0.2238
11	0.1127	18.4535	0.2705	0.223
12	0.1036	18.6423	0.2727	0.2252
13	0.0956	18.6075	0.272	0.2245
14	0.0885	18.3694	0.2704	0.2229
15	0.0827	18.6625	0.2731	0.2256
16	0.0775	18.5981	0.2705	0.223
17	0.0734	18.4934	0.2709	0.2234
18	0.0692	18.2133	0.2688	0.2213
19	0.0661	18.354	0.2683	0.2208
20	0.0631	18.3515	0.2667	0.2192
21	0.06	18.3313	0.2668	0.2194
22	0.0573	18.3806	0.2677	0.2202
23	0.0551	18.2411	0.2668	0.2193
24	0.053	18.2218	0.2668	0.2193
25	0.0511	18.2438	0.2669	0.2194
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
9	1891	0.43331805682859764	0.09908368505565808	0.23139795990715017	0.16680400489137692
5	3382	0.7749770852428964	0.08961282264696849	0.20899057673085028	0.19154334702010314
4	3733	0.8554078826764436	0.08924070419162561	0.2054359148169006	0.19946040288173333
2	4364	1.0	0.07959014300497551	0.19595164987178051	0.21911773395270323
1	4364	1.0	0.07959014300497551	0.19595164987178051	0.21911773395270323
----------------------------------------------------------------------------------------------------
26	0.0497	18.3281	0.2666	0.2191
27	0.0478	18.2191	0.2671	0.2196
28	0.0464	18.0002	0.2667	0.2192
29	0.0446	18.3016	0.2683	0.2208
30	0.0432	17.97	0.2665	0.2191
31	0.0423	18.4138	0.2657	0.2182
32	0.0411	18.2972	0.2668	0.2194
33	0.04	18.1253	0.2661	0.2186
34	0.039	18.2933	0.2676	0.2201
35	0.038	18.2271	0.2673	0.2198
36	0.0372	18.1515	0.2672	0.2197
37	0.0364	17.9505	0.2667	0.2192
38	0.0357	18.2159	0.2673	0.2198
39	0.0348	18.1593	0.2672	0.2197
40	0.0342	18.2819	0.2661	0.2186
41	0.0333	18.212	0.2667	0.2192
42	0.033	18.0825	0.2658	0.2183
43	0.0322	18.0564	0.2654	0.2179
44	0.0316	18.2051	0.2663	0.2188
45	0.0309	18.0988	0.2648	0.2173
46	0.0301	18.1753	0.2657	0.2183
47	0.0299	18.1494	0.2665	0.219
48	0.0294	18.217	0.2646	0.2171
49	0.0287	18.2033	0.2663	0.2188
50	0.0285	18.2493	0.2645	0.217
51	0.0277	18.2053	0.2645	0.217
52	0.0277	18.2356	0.2653	0.2178
53	0.027	18.4232	0.2681	0.2206
54	0.0267	17.7867	0.2629	0.2154
55	0.0263	17.9397	0.2643	0.2169
56	0.026	17.9762	0.2632	0.2157
57	0.0256	18.1359	0.2655	0.218
58	0.0254	17.995	0.2639	0.2164
59	0.0249	18.0848	0.2652	0.2177
60	0.0246	18.0745	0.2632	0.2157
61	0.024	18.2489	0.2653	0.2179
62	0.024	18.0729	0.2642	0.2167
63	0.0237	18.1038	0.2649	0.2174
64	0.0236	18.3779	0.2648	0.2173
65	0.0232	17.9801	0.2632	0.2157
66	0.0227	18.0538	0.2639	0.2164
67	0.0226	18.0325	0.2641	0.2166
68	0.0223	18.1579	0.2654	0.2179
69	0.0221	18.2186	0.2642	0.2167
70	0.0218	18.0889	0.2635	0.216
71	0.0219	18.1815	0.2646	0.2171
72	0.0212	18.1877	0.2652	0.2177
73	0.0212	18.1386	0.2634	0.2159
--------------------------------------------------
MIN: 0.21537189833976503
MAX: 0.26987446542418714
--------------------------------------------------
