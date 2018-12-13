#!/bin/sh

# 12_11_2018
########################################################################################################################

nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 0 -usr 3 -rst 0 > out/12_11_2018/md_1.gs &
nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 0 -usr 5 -rst 0 > out/12_11_2018/md_2.gs &
nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 0 -usr 10 -rst 0 > out/12_11_2018/md_3.gs &
nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 0 -usr 15 -rst 0 > out/12_11_2018/md_4.gs &

nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 1 -usr 3 -rst 3 > out/12_11_2018/md_5.gs &
nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 1 -usr 3 -rst 5 > out/12_11_2018/md_6.gs &
nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 1 -usr 3 -rst 10 > out/12_11_2018/md_7.gs &
nohup python3.6 -u  Main.py -c "Madrid" -m 2 -g 1 -usr 3 -rst 15 > out/12_11_2018/md_8.gs &


