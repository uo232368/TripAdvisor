#!/bin/bash

WHERE="Barcelona"
TEST="DEV"


nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 1 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 1 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 1 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 1 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 0 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 0 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 0 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -imgs 1 -m 2 -g 0 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8_img.gs" &
