Using TensorFlow backend.
[94mObteniendo datos...[0m
[93m[AVISO] 	Usuarios: 33537[0m
[93m[AVISO] 	Restaurantes: 5881[0m
[93m[AVISO] Cargando datos generados previamente...[0m
[94mCreando modelo...[0m


##################################################
 MODELV4
##################################################
 modelv4_deep
##################################################
 Negativos: 10+10
##################################################
[93m[AVISO] Existen 1 combinaciones posibles[0m
--------------------------------------------------
97f7f529e4288ba75253c9a5fb6234df
--------------------------------------------------
[1mlearning_rate: [0m1e-05
--------------------------------------------------
E	T_LOSS	MEAN_POS	PCNT	PCNT-1
1	0.927	23.9141	0.3395	0.292
2	0.7396	22.0021	0.3186	0.2711
3	0.5834	21.2963	0.307	0.2595
4	0.474	20.4785	0.2987	0.2512
5	0.394	20.5252	0.2977	0.2502
6	0.3337	19.7651	0.2887	0.2412
7	0.2875	19.7782	0.287	0.2395
8	0.2506	19.769	0.2849	0.2375
9	0.2204	19.4065	0.2818	0.2343
10	0.196	19.3421	0.2813	0.2338
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
11	0.175	19.2954	0.2794	0.2319
12	0.1577	19.4207	0.2815	0.234
13	0.1432	19.4808	0.2773	0.2298
14	0.1304	19.0637	0.2784	0.2309
15	0.1195	19.2649	0.2795	0.232
16	0.1098	18.8165	0.2775	0.23
17	0.1017	19.2786	0.2777	0.2302
18	0.0939	19.1806	0.2777	0.2302
19	0.0876	19.0752	0.2756	0.2281
20	0.0818	18.8834	0.2743	0.2268
21	0.0762	19.1091	0.2745	0.227
22	0.0715	19.0525	0.276	0.2286
23	0.0671	19.1519	0.2759	0.2284
24	0.0631	18.8685	0.2753	0.2278
25	0.0598	18.5357	0.2746	0.2271
26	0.0564	19.1068	0.2752	0.2277
27	0.0533	18.7461	0.2737	0.2262
28	0.0506	19.0321	0.2754	0.2279
29	0.048	19.0834	0.2745	0.227
30	0.0457	18.6652	0.2719	0.2244
31	0.0436	18.6849	0.2737	0.2262
32	0.0415	18.5804	0.2731	0.2256
33	0.0397	18.5889	0.2742	0.2267
34	0.038	18.8545	0.2739	0.2265
35	0.0363	18.6719	0.2726	0.2252
36	0.0348	18.6577	0.2706	0.2231
37	0.0333	18.5687	0.2703	0.2228
38	0.0321	18.942	0.2734	0.2259
39	0.0306	18.9077	0.2719	0.2244
40	0.0296	18.783	0.2737	0.2262
41	0.0283	18.8352	0.273	0.2255
42	0.0274	18.9292	0.2739	0.2264
43	0.0263	18.8371	0.2733	0.2258
44	0.0252	18.7958	0.2731	0.2256
45	0.0245	18.7452	0.2721	0.2246
46	0.0237	18.5958	0.2717	0.2242
47	0.0229	18.5167	0.2704	0.2229
48	0.0223	18.583	0.2721	0.2246
49	0.0213	19.0202	0.2745	0.227
50	0.0206	18.7722	0.2716	0.2241
51	0.0201	18.5974	0.2681	0.2206
52	0.0194	18.4001	0.2702	0.2227
53	0.0187	18.5758	0.2721	0.2246
54	0.0181	18.8286	0.2725	0.225
55	0.0178	18.3724	0.269	0.2215
56	0.0172	18.8566	0.2725	0.225
57	0.0166	18.3907	0.2696	0.2221
