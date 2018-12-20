#!/bin/sh

# 12_11_2018
########################################################################################################################


WHERE="Madrid"
TEST="DEV"

#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8.gs" &

nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 0 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 0 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 0 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8_img.gs" &