#! /bin/bash  
RUNS=1
ENV="CartPole-v0"
DIR="results"
AIM="reward"
# CUDA=2

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    echo "make a new dir"
    mkdir ${DIR}
fi

for ((i=0;i<${RUNS};i++));
do
    python attack_main.py --no-attack --env-name ${ENV} --resdir ${DIR}/ --no-cuda --run ${i} --seed ${i}
    for RADIUS in 0.1
    do
        for FRAC in 0.3
        do
            for TYPE in wb semirand rand
            do
                python attack_main.py --env-name ${ENV} --resdir ${DIR}/ --no-cuda --radius ${RADIUS} --seed ${i} --run ${i} --frac ${FRAC} --type ${TYPE} --aim ${AIM}
            done
        done
    done
done