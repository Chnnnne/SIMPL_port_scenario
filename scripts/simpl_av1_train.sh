CUDA_VISIBLE_DEVICES=0 python /data/wangchen/SIMPL/train.py \
  --features_dir /private/wangchen/instance_model/instance_model_data_simpl/ \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --val_interval 2 \
  --train_epoches 30 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path config.simpl_cfg
