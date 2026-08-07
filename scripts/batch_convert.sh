INPUT_WILDCARD="/lcrc/group/ATLAS/users/nluongo/ParticleMan/samples/ntup_Run3/*.root"
OUTPUT_DIR="/lcrc/group/ATLAS/users/nluongo/ParticleMan/samples/h5_Run3"
for INPUT_FILE in `ls ${INPUT_WILDCARD}`; do
    BASE_NAME=`echo ${INPUT_FILE} | rev | cut -d "/" -f1 | rev | cut -d "." -f1`
    OUTPUT_FILE="${BASE_NAME}.h5"
    OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILE}"
    echo $INPUT_FILE
    uv run scripts/convert_ntuple_to_h5.py $INPUT_FILE $OUTPUT_PATH
    echo $OUTPUT_PATH
done
