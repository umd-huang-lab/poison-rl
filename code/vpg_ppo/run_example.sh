#! /bin/bash  
RUNS=1
ENV="CartPole-v0" 
LEARNER="vpg" #ppo
DIR="results"
EPS=1000 
STEPS=300 
DEVICE="cpu"
AIM="action" 

if [ -d ${DIR} ]; then
    echo "dir exists"
else
    echo "make a new dir"
    mkdir ${DIR}
fi

for ((i=0;i<${RUNS};i++));
do
    python main.py --no-attack --env ${ENV} --resdir ${DIR}/ --learner ${LEARNER} --episodes ${EPS} --steps ${STEPS} --device ${DEVICE} --run ${i}
    for RADIUS in 0.1
    do
        for FRAC in 0.3
        do
            for TYPE in wb rand semirand
            do
                python main.py --env ${ENV} --resdir ${DIR}/ --learner ${LEARNER} --episodes ${EPS} --steps ${STEPS} --device ${DEVICE} --radius ${RADIUS} --run ${i} --frac ${FRAC} --type ${TYPE} --aim ${AIM}
            done
        done
    done
done