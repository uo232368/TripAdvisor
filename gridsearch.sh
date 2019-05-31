#!/bin/bash

# 20/02/2019
########################################################################################################################

STAGE="grid"

GPU=0

T_NEG="10+10"
L_NEG="n"
RATES="1e-5 5e-6 1e-6"
EPOCHS=100

PLACES=("Madrid")


for WHERE in ${PLACES[*]} ;do

    #nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 0 -stage "$STAGE" -d 1 -lr $RATES -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_TAKE.gs" &
    #GPU=$(($(($GPU+1%2))%2))
    #nohup /usr/bin/python3.6 -u  Main.py -use_imgs 0 -use_like 1 -stage "$STAGE" -d 1 -lr $RATES -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_LIKE.gs" &
    #GPU=$(($(($GPU+1%2))%2))
    #nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 1 -stage "$STAGE" -d 1 -lr $RATES -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_BOTH.gs" &
    #GPU=$(($(($GPU+1%2))%2))

done