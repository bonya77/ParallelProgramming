#!/bin/bash

THREADS=(1 2 4 8 16)
PROGRAMS=(
    "p_variant1Static"
    "p_variant1Dynamic"
    "p_variant1Guided"
    "p_variant2Static"
    "p_variant2Dynamic"
    "p_variant2Guided"
)

OUTPUT_FILE="results.txt"
> $OUTPUT_FILE

for prog in "${PROGRAMS[@]}"; do
    echo "======================================" >> $OUTPUT_FILE
    echo "RUNNING: $prog" >> $OUTPUT_FILE
    echo "======================================" >> $OUTPUT_FILE
    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        echo "Threads: $t" >> $OUTPUT_FILE
        # Запускаем и записываем ТОЛЬКО время
        ./$prog | grep "Time Sequential" >> $OUTPUT_FILE
        echo "" >> $OUTPUT_FILE
    done
    echo "" >> $OUTPUT_FILE
done

echo "Результаты сохранены в $OUTPUT_FILE"