camera_name=corner

python -u metaworld_data.py \
    --num_trail 100 \
    --num_workers 10 \
    --resolution 512 512 \
    --out_state --use_rgb \
    --out_video \
    --camera_name $camera_name \
    --data_dir /data/jaq/metaworld_data/$camera_name

