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
1	0.9108	21.6471	0.315	0.2675
2	0.627	20.173	0.2977	0.2502
3	0.4679	19.6363	0.2901	0.2427
4	0.3493	19.6829	0.2874	0.2399
5	0.2746	19.2906	0.2829	0.2354
6	0.227	19.2963	0.2828	0.2353
7	0.1935	19.1579	0.281	0.2336
8	0.1677	19.2214	0.2803	0.2329
9	0.148	19.1577	0.2778	0.2303
10	0.1328	18.7358	0.2757	0.2282
11	0.1201	18.8655	0.2749	0.2274
12	0.1103	18.7802	0.2753	0.2278
13	0.1017	18.5937	0.2745	0.227
14	0.0937	18.4306	0.2734	0.2259
15	0.0876	18.4645	0.2726	0.2251
16	0.0823	18.5108	0.2747	0.2272
17	0.0776	18.536	0.2736	0.2261
18	0.0732	18.2071	0.2692	0.2217
19	0.0697	18.1256	0.2691	0.2216
20	0.0662	18.1952	0.2694	0.2219
21	0.0634	18.1503	0.2688	0.2214
22	0.0608	18.115	0.2686	0.2211
23	0.0581	18.2351	0.2692	0.2218
24	0.0563	18.2303	0.2692	0.2217
25	0.0539	18.2828	0.2689	0.2215
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
9	1891	0.43331805682859764	0.09893537995342687	0.23124965480491896	0.16695230999360813
5	3382	0.7749770852428964	0.08945703849576361	0.20883479257964543	0.19169913117130802
4	3733	0.8554078826764436	0.08729503208431544	0.20349024270959043	0.2014060749890435
2	4364	1.0	0.0783020554471353	0.19466356231394027	0.22040582151054347
1	4364	1.0	0.0783020554471353	0.19466356231394027	0.22040582151054347
----------------------------------------------------------------------------------------------------
26	0.052	18.1366	0.2679	0.2204
27	0.0502	18.3052	0.2696	0.2222
28	0.0489	18.2709	0.2687	0.2212
29	0.0474	18.1423	0.2688	0.2214
30	0.0458	18.2725	0.269	0.2215
31	0.0447	18.0429	0.2663	0.2188
32	0.0432	18.1118	0.2672	0.2197
33	0.0419	18.1425	0.2686	0.2211
34	0.0412	18.2021	0.2676	0.2201
35	0.0397	18.2793	0.2683	0.2208
36	0.0393	18.0717	0.2664	0.2189
37	0.0382	18.0614	0.266	0.2185
38	0.0374	18.154	0.2675	0.22
39	0.0366	18.0557	0.2655	0.218
40	0.0356	18.2585	0.2669	0.2194
41	0.0351	18.1721	0.267	0.2195
42	0.0342	18.189	0.2665	0.219
43	0.0336	18.1079	0.268	0.2205
44	0.0332	18.1075	0.2663	0.2188
45	0.0322	18.0827	0.2662	0.2187
46	0.032	18.0788	0.2668	0.2193
47	0.0313	17.8767	0.2643	0.2168
48	0.0309	18.1072	0.2659	0.2184
49	0.0305	18.0965	0.2662	0.2187
50	0.0297	18.1464	0.2652	0.2177
51	0.0292	18.3091	0.267	0.2195
52	0.0285	18.2067	0.266	0.2185
53	0.0284	18.1778	0.2663	0.2188
54	0.028	18.0667	0.2642	0.2167
55	0.0274	18.2919	0.2655	0.218
56	0.027	18.1024	0.2654	0.2179
57	0.0268	18.0811	0.2668	0.2193
58	0.0265	18.0465	0.265	0.2175
59	0.0261	18.2062	0.2653	0.2178
60	0.0256	17.9702	0.2634	0.2159
61	0.0254	17.9271	0.265	0.2175
62	0.0251	18.2679	0.2661	0.2186
63	0.0248	18.0878	0.266	0.2185
64	0.0242	18.0543	0.2642	0.2167
65	0.0244	17.9826	0.2634	0.2159
66	0.024	17.9773	0.2631	0.2156
67	0.0235	18.0545	0.2635	0.216
68	0.0232	18.082	0.2637	0.2162
69	0.023	17.9592	0.2645	0.217
70	0.0228	18.0493	0.2642	0.2167
71	0.0225	18.1725	0.2644	0.2169
72	0.0223	17.95	0.2639	0.2164
73	0.022	18.0667	0.2639	0.2164
74	0.0218	18.0623	0.2642	0.2168
75	0.0218	18.0825	0.2633	0.2158
76	0.0212	18.1496	0.2642	0.2167
77	0.0213	18.0593	0.2645	0.217
78	0.021	17.8866	0.2632	0.2157
79	0.0209	18.0566	0.2648	0.2173
80	0.0204	18.0988	0.2659	0.2184
81	0.0205	17.9507	0.2649	0.2174
--------------------------------------------------
MIN: 0.2156170421877046
MAX: 0.2674870687281094
--------------------------------------------------