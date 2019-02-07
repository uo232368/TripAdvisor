#!/bin/sh

# 12_11_2018
########################################################################################################################


WHERE="Barcelona"
TEST="DEV"

nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -lr 1e-7 -m 3 -g 1 -imgs 1 -usr 3 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_IMGS.gs" &

#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -lr 1e-4 1e-5 1e-6 1e-7 -m 3 -g 1 -imgs 1 -usr 3 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_IMGS.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -lr 1e-3 1e-4 1e-5 1e-6 -m 3 -g 1 -imgs 0 -usr 3 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST".gs" &

exit
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 3 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_1.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 5 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_2.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 10 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_3.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 0 -usr 15 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_4.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 3 > "out/16_01_2019/"$WHERE"_"$TEST"_5.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 5 > "out/16_01_2019/"$WHERE"_"$TEST"_6.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 10 > "out/16_01_2019/"$WHERE"_"$TEST"_7.gs" &
#nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -d 0.5 -m 2 -g 1 -usr 3 -rst 15 > "out/16_01_2019/"$WHERE"_"$TEST"_8.gs" &

nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 3 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_1_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 5 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_2_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 10 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_3_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 15 -rst 0 > "out/16_01_2019/"$WHERE"_"$TEST"_4_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 0 -usr 3 -rst 3 > "out/16_01_2019/"$WHERE"_"$TEST"_5_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 1 -usr 3 -rst 5 > "out/16_01_2019/"$WHERE"_"$TEST"_6_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 0 -usr 3 -rst 10 > "out/16_01_2019/"$WHERE"_"$TEST"_7_img.gs" &
nohup /usr/bin/python3.6 -u  Main.py -c $WHERE -imgs 1 -m 2 -g 0 -usr 3 -rst 15 > "out/16_01_2019/"$WHERE"_"$TEST"_8_img.gs" &