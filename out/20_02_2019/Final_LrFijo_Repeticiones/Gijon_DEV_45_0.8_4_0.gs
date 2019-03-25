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
1	1.0014	25.8271	0.315	0.2754
2	0.9505	28.9019	0.3083	0.2687
3	0.7464	27.8668	0.3003	0.2608
4	0.5532	26.4089	0.2916	0.252
5	0.3886	25.528	0.2809	0.2414
6	0.2835	25.8551	0.2873	0.2477
7	0.2208	25.8131	0.2802	0.2406
8	0.1786	25.1612	0.2774	0.2378
9	0.1472	24.1215	0.2783	0.2388
10	0.1267	25.0491	0.2778	0.2382
11	0.1095	24.6262	0.2817	0.2421
12	0.0952	24.2967	0.279	0.2394
13	0.0849	23.7827	0.2698	0.2302
14	0.0762	24.1799	0.2732	0.2336
15	0.0684	23.6425	0.2629	0.2233
16	0.0619	22.9836	0.2682	0.2286
17	0.0575	23.7056	0.2643	0.2248
18	0.0523	22.8341	0.2667	0.2271
19	0.0488	22.8762	0.2686	0.2291
20	0.045	23.5794	0.2708	0.2312
21	0.0409	24.1893	0.2686	0.229
22	0.0388	23.9813	0.2711	0.2315
23	0.037	24.4533	0.269	0.2294
24	0.0351	23.0841	0.2662	0.2266
25	0.0324	23.4813	0.2719	0.2323
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
9	186	0.10095245721355516	0.17291942846162509
5	339	0.06919973301539428	0.1733624626822029
4	376	0.06016701578871639	0.16384414091040633
2	428	0.049340784250711436	0.14757836505244168
1	428	0.049340784250711436	0.14757836505244168
----------------------------------------------------------------------------------------------------
26	0.0306	24.3014	0.2737	0.2341
27	0.0288	23.6402	0.2794	0.2398
28	0.0285	24.3107	0.2714	0.2318
29	0.0265	24.8131	0.2743	0.2348
30	0.0262	23.4673	0.2691	0.2296
31	0.0243	23.9089	0.2729	0.2333
32	0.0229	24.8061	0.2706	0.231
33	0.0233	24.6121	0.271	0.2314
34	0.0217	23.757	0.2777	0.2381
35	0.0208	23.9346	0.2708	0.2313
36	0.0202	23.6308	0.2642	0.2247
37	0.0192	24.6332	0.2688	0.2293
38	0.0189	23.7126	0.2637	0.2242
39	0.0181	24.2313	0.2672	0.2276
40	0.0177	24.2967	0.2699	0.2303
41	0.0172	24.2664	0.2696	0.23
42	0.0171	24.493	0.2713	0.2317
43	0.0158	23.5444	0.2683	0.2288
44	0.0156	23.7009	0.2695	0.23
45	0.0151	22.6916	0.2642	0.2246
46	0.0153	24.4182	0.2658	0.2263
47	0.0145	22.8902	0.2602	0.2207
48	0.014	25.0864	0.2739	0.2344
49	0.0138	23.9322	0.2695	0.2299
50	0.0138	22.6986	0.2585	0.2189
51	0.013	22.9486	0.2622	0.2226
52	0.0131	22.2266	0.2607	0.2211
53	0.0126	22.3131	0.2671	0.2276
54	0.0124	21.4907	0.2618	0.2222
55	0.0118	20.7804	0.2594	0.2199
56	0.0121	22.4743	0.262	0.2225
57	0.012	22.2664	0.2653	0.2257
58	0.0115	23.229	0.2639	0.2243
59	0.0108	21.8832	0.2636	0.2241
60	0.0111	21.9883	0.2618	0.2222
61	0.0114	22.1472	0.2652	0.2256
62	0.0102	21.7173	0.2609	0.2213
63	0.0108	22.8014	0.2646	0.225
64	0.0102	22.7687	0.2694	0.2299
65	0.0101	23.7453	0.2682	0.2287
--------------------------------------------------
MIN: 0.21891266858111097
MAX: 0.27542093457516154
--------------------------------------------------
