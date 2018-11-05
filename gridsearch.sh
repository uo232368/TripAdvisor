#!/bin/sh

# 04/10/2018
########################################################################################################################

#nohup python3.6 -u  Main.py -m 2 -g 0 -d 0.5 -over "none" > out/model2/model2_noover_dpout.gs &
#nohup python3.6 -u  Main.py -m 2 -g 1 -d 0.0 -over "none" > out/model2/model2_noover_nodpout.gs &
#nohup python3.6 -u  Main.py -m 2 -g 0 -d 0.5 -over "2" > out/model2/model2_over_dpout.gs &
#nohup python3.6 -u  Main.py -m 2 -g 1 -d 0.0 -over "2" > out/model2/model2_over_nodpout.gs &

#nohup python3.6 -u  Main.py -m 3 -g 0 -d 0.5 -over "none" > out/model3/model3_noover_dpout.gs &
#nohup python3.6 -u  Main.py -m 3 -g 1 -d 0.0 -over "none" > out/model3/model3_noover_nodpout.gs &
#nohup python3.6 -u  Main.py -m 3 -g 0 -d 0.5 -over "2" > out/model3/model3_over_dpout.gs &
#nohup python3.6 -u  Main.py -m 3 -g 1 -d 0.0 -over "2" > out/model3/model3_over_nodpout.gs &

# 29/10/2018
########################################################################################################################

nohup python3.6 -u  Main.py -m 1 -g 0 -d 1.0 -top 5 -lr 1e-4 1e-5 1e-6 -emb 512 256 > out/29_10_2018/model1_sigmoid_p1.gs &
nohup python3.6 -u  Main.py -m 1 -g 0 -d 1.0 -top 5 -lr 1e-7 1e-8 1e-9 -emb 512 256 > out/29_10_2018/model1_sigmoid_p2.gs &

nohup python3.6 -u  Main.py -m 1 -g 1 -d 1.0 -top 5 -lr 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 -emb 128 > out/29_10_2018/model1_sigmoid_p3.gs &
nohup python3.6 -u  Main.py -m 1 -g 1 -d 1.0 -top 5 -lr 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 -emb 1024 > out/29_10_2018/model1_sigmoid_p4.gs &
