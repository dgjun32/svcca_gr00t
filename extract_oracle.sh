CUDA_VISIBLE_DEVICES=0 python extract_layer_features.py --dataset_path /home/dongjun/gr00t_analysis/svcca_dataset/segmented_depth_1_trash_separation_3_objs_unseen_251126_pnp_coke_to_plastic_pick_split \
 --model_path /home/dongjun/gr00t_analysis/checkpoints/gr00t_3_objs_separation_seen_unseen_depth_1_fullft/checkpoint-15000 \
 --cache_path /home/dongjun/gr00t_analysis/cached_features \


CUDA_VISIBLE_DEVICES=0 python extract_layer_features.py --dataset_path /home/dongjun/gr00t_analysis/svcca_dataset/segmented_depth_1_trash_separation_3_objs_unseen_251126_pnp_coke_to_plastic_place_split \
 --model_path /home/dongjun/gr00t_analysis/checkpoints/gr00t_3_objs_separation_seen_unseen_depth_1_fullft/checkpoint-15000 \
 --cache_path /home/dongjun/gr00t_analysis/cached_features \