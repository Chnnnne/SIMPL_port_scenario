echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir /data/wangchen/dataset/argo/val/data/ \
  --save_dir data_argo/features/ \
  --small
  # --debug --viz

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir /data/wangchen/dataset/argo/train/data/ \
  --save_dir data_argo/features/ \
  --small

# echo "-- Processing test set..."
# python data_argo/run_preprocess.py --mode test \
#   --data_dir /data/wangchen/dataset/argo/test_obs/data/ \
#   --save_dir data_argo/features/ \
#   --small