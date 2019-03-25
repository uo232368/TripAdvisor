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
1	0.9116	21.6318	0.3116	0.2641
2	0.6316	20.2709	0.2951	0.2476
3	0.4554	20.0868	0.2894	0.2419
4	0.3324	19.6556	0.2865	0.2391
5	0.2627	19.2706	0.282	0.2345
6	0.2168	19.1033	0.278	0.2305
7	0.1851	19.0951	0.2765	0.229
8	0.1607	19.2745	0.278	0.2305
9	0.1426	19.0648	0.2746	0.2271
10	0.1272	19.0057	0.2746	0.2271
11	0.1158	19.036	0.2737	0.2262
12	0.106	19.0034	0.2742	0.2267
13	0.0979	18.7704	0.2727	0.2252
14	0.0911	18.8396	0.2718	0.2243
15	0.0848	18.8811	0.2713	0.2238
16	0.0791	19.115	0.2723	0.2248
17	0.0753	18.8847	0.2699	0.2224
18	0.0712	18.7204	0.2714	0.2239
19	0.0676	19.129	0.2727	0.2252
20	0.0641	18.9244	0.2704	0.223
21	0.0615	18.841	0.2686	0.2212
22	0.0586	18.6838	0.2714	0.2239
23	0.0561	18.8162	0.2708	0.2233
24	0.0543	18.86	0.2694	0.2219
25	0.0522	18.6769	0.2691	0.2216
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
9	1891	0.43331805682859764	0.091624596627625	0.2239388714791171	0.17426309331941
5	3382	0.7749770852428964	0.08571686973757006	0.20509462382145188	0.19543929992950157
4	3733	0.8554078826764436	0.08415522060133647	0.20035043122661145	0.2045458864720225
2	4364	1.0	0.07573833896467559	0.19209984583148057	0.22296953799300315
1	4364	1.0	0.07573833896467559	0.19209984583148057	0.22296953799300315
----------------------------------------------------------------------------------------------------
26	0.0503	18.6267	0.2705	0.223
27	0.0489	18.5415	0.2682	0.2207
28	0.0472	18.6146	0.2683	0.2208
29	0.0456	18.632	0.2673	0.2198
30	0.0446	18.8013	0.2684	0.2209
31	0.0432	18.4003	0.2669	0.2194
32	0.0422	18.5871	0.2661	0.2186
33	0.0407	18.6748	0.2663	0.2188
34	0.04	18.7722	0.2665	0.219
35	0.0391	18.4569	0.2661	0.2186
36	0.0377	18.8059	0.2672	0.2197
37	0.0369	18.6091	0.2667	0.2192
38	0.0363	18.3396	0.2648	0.2173
39	0.0355	18.324	0.2648	0.2173
40	0.0347	18.5726	0.2669	0.2194
41	0.0341	18.4349	0.2668	0.2194
42	0.0333	18.7729	0.2665	0.219
43	0.0327	18.4631	0.2658	0.2183
44	0.0321	18.4879	0.2666	0.2191
45	0.0314	18.2589	0.2652	0.2177
46	0.0311	18.3327	0.2661	0.2186
47	0.0304	18.3994	0.2667	0.2192
48	0.0298	18.629	0.2676	0.2201
49	0.0293	18.3139	0.2654	0.2179
50	0.0291	18.3522	0.2664	0.2189
51	0.0284	18.4716	0.2657	0.2182
52	0.0278	18.5591	0.2665	0.219
53	0.0278	18.6599	0.2672	0.2197
--------------------------------------------------
MIN: 0.21730191015964154
MAX: 0.2641071261461617
--------------------------------------------------
