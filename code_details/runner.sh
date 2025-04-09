# config_names=("obb_base_y8x" "obb_base_y11l" "obb_base_y11m" "obb_base")  # Add your config names here
config_names=("obb_base_y11m")
# Mode selection
mode="to_geojson" # train, predict, to_geojson
train_folder="" # train data
val_folder="" # val data
predict_region="" #"Predict data"

# Loop over config names
for i in "${!config_names[@]}"; do
  config_name="${config_names[$i]}"
  gpu_id=$i  # Assign each config to a separate GPU

  export CUDA_VISIBLE_DEVICES=$gpu_id

  if [ "$mode" = "train" ]; then
    declare -A args=(
      [train_folder]=$train_folder
      [val_folder]=$val_folder
      [config_name]=$config_name
    )
  elif [ "$mode" = "predict" ]; then
    declare -A args=(
      [train_folder]=$train_folder
      [val_folder]=$val_folder
      [config_name]=$config_name
      [z_predict_region]=$predict_region
    )
  elif [ "$mode" = "to_geojson" ]; then
    declare -A args=(
      [train_folder]=$train_folder
      [val_folder]=$val_folder
      [config_name]=$config_name
      [z_predict_region]=$predict_region
    )
  fi

  # Prepare sorted arguments and command
  sorted_keys=$(for key in "${!args[@]}"; do echo "$key=${args[$key]}"; done | sort)
  args=$(echo "$sorted_keys" | paste -sd '&' -)
  save_folder="runs/obb/$args"

  cmd="python -u runner.py $mode $save_folder"
  mkdir -p "$save_folder"

  # Start the process in the background
  echo "Starting $mode for $config_name on GPU $gpu_id"
  nohup $cmd > "$save_folder/run_${mode}.log" 2>&1 &

done
