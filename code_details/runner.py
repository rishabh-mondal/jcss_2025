import sys
from os.path import join
from time import time
from git import Repo
import importlib

# class
class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value


# Collect mode and UID
mode = sys.argv[1]
uid = sys.argv[2]

# Parse arguments
task = uid.split("/", 2)[1]
args_str = uid.split("/", 2)[-1]
args = DotDict([arg.split("=") for arg in args_str.split("&")])

repo = Repo(search_parent_directories=True)
repo_root_dir = repo.git.rev_parse("--show-toplevel")

# config
config_dir = join(repo_root_dir, "configs")
config = importlib.import_module(f"configs.{args.config_name}")

start_time = time()
if mode in ["train", "predict"]:
    # imports
    import tempfile
    from ultralytics import YOLO

    # train val directories
    base_dir = join(repo_root_dir, "data")
    train_image_dir = join(base_dir, args.train_folder, "images")
    val_image_dir = join(base_dir, args.val_folder, "images")

    temp_file = tempfile.NamedTemporaryFile(suffix=".yml")
    with open(temp_file.name, "w") as f:
        f.write(
            f"""train: {train_image_dir}
val: {val_image_dir}
nc: {len(config.classes)}
names: {config.classes}"""
        )
    if mode == "train":
        print("Training on", train_image_dir)
        print("Validating on", val_image_dir)
        model = YOLO(model=config.model, task=task)
        model.train(data=temp_file.name, name=args_str, **config.train_args)
    elif mode == "predict":
        train_args_str = "&".join(args_str.split("&")[:-1])
        model_path = join(repo_root_dir, "runs", task, train_args_str, "weights", "best.pt")
        model = YOLO(model=model_path, task=task)
        image_dir = join(repo_root_dir, "data", args.z_predict_region, "images")
        print("Predicting on", image_dir)
        model.predict(image_dir, stream=False, name=args_str, **config.predict_args)
    else:
        raise ValueError(f"Mode {mode} not supported")
elif mode == "to_geojson":
    from run import ultralytics_to_geojson

    prediction_dir = join(repo_root_dir, "runs", task, args_str)
    print(f"{config.to_geojson_args=}")

    ultralytics_to_geojson(
        image_dir=join(repo_root_dir, "data", args.z_predict_region, "images"),
        prediction_dir=prediction_dir,
        target_geojson_path=join(repo_root_dir, "regions", "labels", f"{args.z_predict_region}.geojson"),
        metadata_path=join(repo_root_dir, "data", args.z_predict_region, "metadata.geojson"),
        task=task,
        classes=config.classes,
        conf_threshold=config.to_geojson_args["conf_threshold"],
        save_dir=prediction_dir,
        nms_iou=config.to_geojson_args["nms_iou"],
    )
else:
    raise ValueError(f"Mode {mode} not supported")
end_time = time()

seconds = end_time - start_time
minutes = seconds / 60

print(f"Running took {minutes} minutes")
