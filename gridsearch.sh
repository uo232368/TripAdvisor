#!/bin/sh

nohup python3.6 -u  Main.py -m 2 -g 0 -d 0.5 -over "none" > out/model2/model2_noover_dpout.gs &
nohup python3.6 -u  Main.py -m 2 -g 1 -d 0.0 -over "none" > out/model2/model2_noover_nodpout.gs &
nohup python3.6 -u  Main.py -m 2 -g 0 -d 0.5 -over "2" > out/model2/model2_over_dpout.gs &
nohup python3.6 -u  Main.py -m 2 -g 1 -d 0.0 -over "2" > out/model2/model2_over_nodpout.gs &

nohup python3.6 -u  Main.py -m 3 -g 0 -d 0.5 -over "none" > out/model3/model3_noover_dpout.gs &
nohup python3.6 -u  Main.py -m 3 -g 1 -d 0.0 -over "none" > out/model3/model3_noover_nodpout.gs &
nohup python3.6 -u  Main.py -m 3 -g 0 -d 0.5 -over "2" > out/model3/model3_over_dpout.gs &
nohup python3.6 -u  Main.py -m 3 -g 1 -d 0.0 -over "2" > out/model3/model3_over_nodpout.gs &
