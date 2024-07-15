CUDA_VISIBLE_DEVICES=0 python /data/wangchen/SIMPL/evaluation.py \
  --features_dir /private/wangchen/instance_model/instance_model_data_simpl_test_latency/ \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --use_cuda \
  --adv_cfg_path config.simpl_cfg \
  --model_path /data/wangchen/SIMPL/scripts/saved_models/20240702-233322_Simpl_best.tar