Using TensorFlow backend.
[94mObteniendo datos...[0m
[93m[AVISO] 	Usuarios: 5139[0m
[93m[AVISO] 	Restaurantes: 598[0m
[93m[AVISO] Cargando datos generados previamente...[0m
[94mCreando modelo...[0m


##################################################
 MODELV4
##################################################
 modelv4d2
##################################################
[93m[AVISO] Existen 1 combinaciones posibles[0m
--------------------------------------------------
15d692719045548b880be3c9109b1159
--------------------------------------------------
[1mlearning_rate: [0m0.0001
[1mdropout: [0m0.8
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	1.003	27.4836	0.3159	0.2763
2	0.9577	28.1425	0.3042	0.2646
3	0.76	27.6986	0.2969	0.2573
4	0.5767	26.5888	0.2949	0.2554
5	0.4167	24.5444	0.2817	0.2421
6	0.3042	23.7336	0.2751	0.2356
7	0.2329	24.1028	0.2751	0.2355
8	0.1878	23.2921	0.2682	0.2286
9	0.1555	23.3925	0.2701	0.2306
10	0.1296	23.0514	0.2685	0.2289
11	0.1125	24.1308	0.275	0.2354
12	0.0968	23.8668	0.271	0.2314
13	0.0868	22.2547	0.2595	0.22
14	0.0778	22.9393	0.2611	0.2216
15	0.0698	21.9065	0.2603	0.2208
16	0.0642	21.465	0.2611	0.2215
17	0.0576	23.264	0.2653	0.2257
18	0.0531	23.4065	0.2643	0.2247
19	0.0482	22.8505	0.2605	0.2209
20	0.0447	23.0958	0.2672	0.2277
21	0.042	22.8037	0.2616	0.222
22	0.0395	22.8785	0.2647	0.2252
23	0.0366	24.0304	0.266	0.2264
24	0.0352	23.2593	0.2607	0.2211
25	0.033	23.6145	0.2609	0.2213
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
N_FOTOS_TRAIN (>=)	N_ITEMS	RND-MOD AC	CNT-MOD AC
9	186	0.09738468373429339	0.16935165498236335
5	339	0.08164542134855218	0.1858081510153608
4	376	0.07433533581813691	0.17801246093982684
2	428	0.0618857502319131	0.16012333103364335
1	428	0.0618857502319131	0.16012333103364335
----------------------------------------------------------------------------------------------------
26	0.0314	22.8808	0.2612	0.2216
27	0.0294	23.3762	0.2635	0.224
28	0.0283	24.0935	0.2656	0.226
29	0.0265	24.8738	0.2648	0.2252
30	0.0254	23.8107	0.2664	0.2268
31	0.0247	24.0748	0.2621	0.2225
32	0.0235	23.8598	0.2618	0.2222
33	0.023	24.486	0.2643	0.2248
34	0.022	23.5444	0.2585	0.2189
35	0.0207	23.1168	0.257	0.2174
36	0.0206	23.0327	0.2627	0.2232
37	0.0197	23.8364	0.2672	0.2276
38	0.0192	23.3598	0.2645	0.2249
39	0.0183	23.4369	0.2645	0.2249
40	0.0182	23.521	0.2642	0.2246
41	0.017	23.1752	0.261	0.2215
42	0.0163	24.0678	0.2615	0.2219
43	0.0163	23.0093	0.2569	0.2173
44	0.0158	23.979	0.2643	0.2248
45	0.0157	23.6893	0.2585	0.2189
46	0.0147	23.8458	0.2605	0.2209
47	0.0145	22.1659	0.2466	0.207
48	0.0146	23.7103	0.2577	0.2182
49	0.0135	23.7336	0.2669	0.2273
50	0.0133	24.472	0.2625	0.2229
51	0.0139	23.9907	0.2686	0.229
52	0.0133	23.4136	0.2662	0.2267
--------------------------------------------------
MIN: 0.2070253857037409
MAX: 0.27631983725776726
--------------------------------------------------
