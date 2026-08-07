MAX_JOBS=5
INPUT_WILDCARD="/lcrc/group/ATLAS/users/nluongo/ParticleMan/samples/ntup_Run3/*.root"
OUTPUT_DIR="/lcrc/group/ATLAS/users/nluongo/ParticleMan/samples/h5_Run3"
for INPUT_FILE in `ls ${INPUT_WILDCARD}`; do
    BASE_NAME=`echo ${INPUT_FILE} | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
    OUTPUT_FILE="${BASE_NAME}.h5"
    OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"
    LOG_OUTPUT_FILE="${BASE_NAME}.txt"
    LOG_OUTPUT_PATH="${OUTPUT_DIR}/${LOG_OUTPUT_FILE}"
    echo $INPUT_FILE
    uv run scripts/convert_ntuple_to_h5.py $INPUT_FILE $OUTPUT_PATH > $LOG_OUTPUT_PATH 2>&1 &
    echo $OUTPUT_PATH

    while [ "$(pgrep -c -f "python3 scripts/convert_ntuple_to_h5")" -ge "$MAX_JOBS" ]; do
        wait -n
    done
done

wait
