


PYTHONPATH=. python tools/run_inference.py   --cfg config/inference_config/ace_0.6b_1024.yaml   --instruction "place bed,cabinet,table,chair,painting,rug,lamp,cushion,pillow,ottoman and blanket in this bedroom"   --seed 2024   --output_h 1024   --output_w 1024   --input_path datasets/validation_data/bedroom/images   --save_path experiments/FurniFlip139K/vs-bedroom-v1/exp3/inference_test_run


PYTHONPATH=. nohup python tools/run_train.py --cfg config/train_config/ace_0.6b_1024_train.yaml > experiments/FurniFlip139K/vs-bedroom-v1/exp1.log 2>&1 &