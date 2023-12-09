#LOOP through oprtions in bash code and run the python code

MODE="admm auglag penalty"
RUNS=10

for run in $(seq 1 $RUNS); do
    for mode in $MODE; do
        echo "Running $mode"
        python3 opt_main.py --opt_mode $mode --model resnet --dataset cifar10
    done
done
