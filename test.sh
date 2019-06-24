#!/bin/bash

GPU=1
T_NEG="10+10"
L_NEG="0"
EPOCHS=100
STAGE="test"

WHERE="Gijon"
#RATE="1e-5"
RATE="1e-3"

nohup /usr/bin/python3.6 -u  Main.py -use_imgs 0 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_LIKE.txt" &

#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 0 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_TAKE.txt" &

#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_BOTH.txt" &


#WHERE="Barcelona"
WHERE="Madrid"
RATE="5e-6"
#RATE="1e-5"

#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 0 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_LIKE.txt" &

#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 0 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_TAKE.txt" &

#GPU=$(($(($GPU+1%2))%2))
#nohup /usr/bin/python3.6 -u  Main.py -use_imgs 1 -use_like 1 -stage "$STAGE" -d 1 -lr $RATE -lrdcay "linear_cosine" -e $EPOCHS -c $WHERE -m 6 -g $GPU -nimg "$T_NEG" -nlike "$L_NEG" > "out/24_04_2019/"$WHERE"_"$STAGE"_"$L_NEG"_BOTH.txt" &


