#!/bin/bash

GPU=1
T_NEG="10+10"
L_NEG="n"
EPOCHS=100
STAGE="test"


WHERE="Gijon"

#RATE="1e-5"
#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 0 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_LIKE.gs" &

#RATE="5e-5"
#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 0 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_TAKE.gs" &

RATE="5e-6"
GPU=$(($(($GPU+1%2))%2))
nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_BOTH.gs" &


exit


WHERE="Barcelona"
RATE="1e-6"
nohup /usr/bin/python3.6 -u  Main.py -use_imgs 0 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_LIKE.gs" &

RATE="5e-6"
nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 0 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_TAKE.gs" &

RATE="1e-6"
GPU=$(($(($GPU+1%2))%2))
nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_BOTH.gs" &



WHERE="Madrid"
RATE=""
