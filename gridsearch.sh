#!/bin/sh

# 12_11_2018
########################################################################################################################

nohup python3.6 -u  Main.py -m 1 -g 0 -d 0.5 -lr 1e-4  > out/12_11_2018/model1_t1.gs &
nohup python3.6 -u  Main.py -m 1 -g 0 -d 0.5 -lr 1e-3  > out/12_11_2018/model1_t2.gs &
nohup python3.6 -u  Main.py -m 1 -g 0 -d 0.5 -lr 1e-2  > out/12_11_2018/model1_t3.gs &
nohup python3.6 -u  Main.py -m 1 -g 1 -d 1 -lr 1e-4  > out/12_11_2018/model1_t4.gs &
nohup python3.6 -u  Main.py -m 1 -g 1 -d 1 -lr 1e-3  > out/12_11_2018/model1_t5.gs &
nohup python3.6 -u  Main.py -m 1 -g 1 -d 1 -lr 1e-2  > out/12_11_2018/model1_t6.gs &
