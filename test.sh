#!/bin/bash

RPT=( 0 1 2 3 4 )
GPU=0


WHERE="Gijon"
#nohup /usr/bin/python3.6 -u  Main.py -e 30 -lr 1e-4 -d 0.8 -c "$WHERE" -m 45 -g 0 -pref "10+10" > "out/20_02_2019/"$WHERE"_TEST.tst" &

#for RP in ${RPT[*]} ;do
#    nohup /usr/bin/python3.6 -u  Main.py -e 45 -stage "test" -lrdcay "linear_cosine" -e 100 -lr 1e-4 -d 0.8 -c "$WHERE" -m 45 -g $GPU -pref "10+10" > "out/20_02_2019/"$WHERE"_TEST_"$RP".tst" &
#    GPU=$(($(($GPU+1%2))%2))
#done



WHERE="Barcelona"
#nohup /usr/bin/python3.6 -u  Main.py -e 22 -lr 1e-4 -d 0.8 -c "$WHERE" -m 45 -g 1 -pref "10+10" > "out/20_02_2019/"$WHERE"_TEST.tst" &

#for RP in ${RPT[*]} ;do
#    nohup /usr/bin/python3.6 -u  Main.py -e 45 -stage "test" -lrdcay "linear_cosine" -e 100 -lr 1e-4 -d 0.8 -c "$WHERE" -m 45 -g $GPU -pref "10+10" > "out/20_02_2019/"$WHERE"_TEST_"$RP".tst" &
#    GPU=$(($(($GPU+1%2))%2))
#done


WHERE="Madrid"
#nohup /usr/bin/python3.6 -u  Main.py -e 144 -lr 1e-5 -d 0.8 -c "$WHERE" -m 45 -g 0 -pref "10+10" > "out/20_02_2019/"$WHERE"_TEST.tst" &


#for RP in ${RPT[*]} ;do
#    nohup /usr/bin/python3.6 -u  Main.py -e 45 -stage "test" -lrdcay "linear_cosine" -e 100 -lr 1e-4 -d 0.8 -c "$WHERE" -m 45 -g $GPU -pref "10+10" > "out/20_02_2019/"$WHERE"_TEST_"$RP".tst" &
#    GPU=$(($(($GPU+1%2))%2))
#done