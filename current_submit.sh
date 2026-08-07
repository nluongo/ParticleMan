qsub -l select=1:ngpus=2 -v EPOCHS=80 scripts/pbs/submit_training.pbs
