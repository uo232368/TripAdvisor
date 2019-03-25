#!/bin/bash

# 20/02/2019
########################################################################################################################

TEST="DEV"

GPU=0
NEG="10+10"
RATES="1e-3"

PLACES=("Madrid")
DPOUTS=( 0.8 )
MODELS=( 45 )
RPT=(  )


for WHERE in ${PLACES[*]} ;do
    for DP in ${DPOUTS[*]} ;do
        for MODEL in ${MODELS[*]} ;do
            for RP in ${RPT[*]} ;do

                #echo "$WHERE"_"$TEST"_"$MODEL"_"$DP"_"$RATES"
                nohup /usr/bin/python3.6 -u  Main.py -stage "grid" -d $DP -lr $RATES -lrdcay "linear_cosine" -e 100 -c $WHERE -m $MODEL -g $GPU -pref "$NEG" > "out/20_02_2019/"$WHERE"_"$TEST"_"$MODEL"_"$DP"_"$RATES"_"$RP".gs" &

                GPU=$(($(($GPU+1%2))%2))
           done
        done
    done
done