CUDA_VISIBLE_DEVICES=0 python extract_layer_features.py --dataset_path /home/dongjun/gr00t_svcca/dataset/trash_seperation_critical_moments \
 --model_path /home/dongjun/checkpoints/gr00t_3_objs_separation_seen_unseen_depth_1_fullft/checkpoint-10000 \
 --cache_path /home/dongjun/gr00t_svcca/cached_features \


CUDA_VISIBLE_DEVICES=0 python extract_layer_features.py --dataset_path /home/dongjun/gr00t_svcca/dataset/trash_seperation_critical_moments \
 --model_path /home/dongjun/checkpoints/gr00t_3_objs_separation_seen_depth_1_fullft/checkpoint-10000 \
 --cache_path /home/dongjun/gr00t_svcca/cached_features \