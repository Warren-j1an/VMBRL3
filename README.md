# VMBRL3
Run video pre-training (if you want to run pre-training by yourself):

```
python vmbrl3_pretraining/train_video.py --logdir ~/logdir/pretrain/vmbrl3/rlbench --load_logdir ~/data/pretrain/rlbench --configs metaworld_pretrain
```

Run Meta-world experiments:

```
python vmbrl3_finetuning/train.py --logdir ~/logdir/metaworld_lever_pull/vmbrl3/1 --load_logdir ~/logdir/pretrain/vmbrl3/rlbench --configs metaworld --task metaworld_lever_pull --seed 13
```

Run DeepMind Control Suite experiments:

```
python vmbrl3_finetuning/train.py --logdir ~/logdir/dmc_quadruped_walk/vmbrl3/1 --load_logdir ~/logdir/pretrain/vmbrl3/rlbench --configs dmc_vision --task dmc_quadruped_walk --seed 13
```

## Tips
- Also see the tips available in [DreamerV2 repository](https://github.com/danijar/dreamerv2/blob/main/README.md#tips).