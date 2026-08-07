qsub -l select=1:ngpus=8 \
     -v CONFIG=bbllv08.yaml,EXPERIMENT_NAME=ParticleMan_bbllv08,MAX_EVENTS=10000000,EPOCHS=10 \
    scripts/pbs/submit_training.pbs
