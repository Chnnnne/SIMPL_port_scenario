CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node 2 /data/wangchen/SIMPL/train_ddp.py \
  --features_dir /private/wangchen/instance_model/instance_model_data_simpl/ \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --val_interval 2 \
  --train_epoches 50 \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path config.simpl_cfg
