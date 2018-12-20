#!/bin/bash

TEST="TEST"
DPOUT=0.5 
MODEL=2

WHERE="Barcelona"


EPOCHS=(7 5 9 8 7 8 52 45)

nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[0]} -d $DPOUT -m $MODEL -g 0 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[1]} -d $DPOUT -m $MODEL -g 0 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[2]} -d $DPOUT -m $MODEL -g 0 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[3]} -d $DPOUT -m $MODEL -g 0 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[4]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[5]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[6]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[7]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8.gs" &


EPOCHS=(6 5 9 6 6 6 6 7)

nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[0]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[1]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[2]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[3]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[4]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[5]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[6]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7_img.gs" &
nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[7]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8_img.gs" &

#########################################################################################################################
#
#WHERE="Madrid"
#EPOCHS=(8 7 10 8 31 31 34 41)
#
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[0]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[1]} -d $DPOUT -m $MODEL -g 1 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[2]} -d $DPOUT -m $MODEL -g 1 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[3]} -d $DPOUT -m $MODEL -g 1 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[4]} -d $DPOUT -m $MODEL -g 0 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[5]} -d $DPOUT -m $MODEL -g 0 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[6]} -d $DPOUT -m $MODEL -g 0 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[7]} -d $DPOUT -m $MODEL -g 0 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8.gs" &
#
#
#EPOCHS=(6 5 6 8 6 6 7 8)
#
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[0]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[1]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[2]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[3]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 15 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_4_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[4]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[5]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[6]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[7]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8_img.gs" &

########################################################################################################################

#WHERE="Gijon"
#
#EPOCHS=(13 18 17 13 13 13 15)
#
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[0]} -d $DPOUT -m $MODEL -g 0 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[1]} -d $DPOUT -m $MODEL -g 0 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[2]} -d $DPOUT -m $MODEL -g 0 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[3]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[4]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[5]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[6]} -d $DPOUT -m $MODEL -g 1 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8.gs" &
#
#
#EPOCHS=(12 17 18 13 15 13 13)
#
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[0]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 3 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_1_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[1]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 5 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_2_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[2]} -d $DPOUT -imgs 1 -m $MODEL -g 1 -usr 10 -rst 0 > "out/17_12_2018/"$WHERE"_"$TEST"_3_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[3]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 3 > "out/17_12_2018/"$WHERE"_"$TEST"_5_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[4]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 5 > "out/17_12_2018/"$WHERE"_"$TEST"_6_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[5]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 10 > "out/17_12_2018/"$WHERE"_"$TEST"_7_img.gs" &
#nohup python3.6 -u  Main.py -c $WHERE -e ${EPOCHS[6]} -d $DPOUT -imgs 1 -m $MODEL -g 0 -usr 3 -rst 15 > "out/17_12_2018/"$WHERE"_"$TEST"_8_img.gs" &