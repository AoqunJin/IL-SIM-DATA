camera_name=corner

python -u metaworld_data.py \
--num_trail 5 \
--num_workers 10 \
--resolution 1024 1024 \
--camera_name $camera_name \
--data_dir metaworld_data/$camera_name
