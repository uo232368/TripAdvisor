#!/bin/bash

# 20/02/2019
########################################################################################################################

STAGE="grid"

GPU=0

T_NEG="10+10"
L_NEG="0"
RATES="1e-1 5e-2 1e-2 5e-2 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6 1e-6"
#RATES="1e-5 5e-6 1e-6"

EPOCHS=100

PLACES=("Gijon")


for WHERE in ${PLACES[*]} ;do

    nohup /usr/bin/python3.6 -u  Main.py -use_imgs 0 -use_like 1 -stage "$STAGE" -d 1 -lr $RATES -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_LIKE.txt" &
    #GPU=$(($(($GPU+1%2))%2))
    #nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 0 -stage "$STAGE" -d 1 -lr $RATES -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_TAKE.txt" &
    #GPU=$(($(($GPU+1%2))%2))
    #nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 1 -stage "$STAGE" -d 1 -lr $RATES -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_BOTH.txt" &
    #GPU=$(($(($GPU+1%2))%2))

done