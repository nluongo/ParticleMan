uv run scripts/train.py --config-name=bbllv08 \
    data=bbllv08_classify data.max_events=100000 \
    training.pretrained_checkpoint=/lcrc/group/ATLAS/users/nluongo/ParticleMan/checkpoints/best_model_epoch_110.pt \
    output=noop
