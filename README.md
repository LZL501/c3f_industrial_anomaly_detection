Memory-based Coarse-to-Fine Feature Fusion for Industrial Anomaly Detection
==========================================================================

This is the official PyTorch implementation of the paper:
"Anomaly or Characteristic: Memory-based Coarse-to-Fine Feature Fusion for Industrial Anomaly Detection".


Dataset
-------
We evaluate our method on two widely-used benchmarks:

1. MVTec-AD: A high-resolution dataset for industrial visual anomaly detection.  
   Download: https://www.mvtec.com/company/research/datasets/mvtec-ad

2. VisA: A large-scale dataset for industrial anomaly detection in fine-grained textures and objects.  
   Download: https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

Training
--------
To train the model on a specific VisA category (e.g., `candle`), run:

   python main.py --base configs/visa/82/visa_candle.yaml --gpus 0

You can modify the YAML config path to switch between categories or datasets (e.g., MVTec-AD).

