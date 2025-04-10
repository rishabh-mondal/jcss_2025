model = "yolov8l-obb.pt"
image_size = 640
epochs = 1000
train_args = {
    "imgsz": image_size,
    "save_period": 10,
    "epochs": epochs,
    "patience": 100,
    "val": True,
    "val_interval": 10,
    "batch": 64,
    "exist_ok": True,
    "save": True,
    "workers": 0,
    "verbose": True,
}

predict_args = {
    "imgsz": image_size,
    "conf": 0.001,
    "iou": 0.5,
    "exist_ok": True,
    "save_txt": True,
    "save_conf": True,
    "save_crop": True,  # obb does not support this
    "verbose": True,
}

to_geojson_args = {
    "conf_threshold": 0.25,
    "nms_iou": 0.5,
}

classes = ["CFCBK", "FCBK", "Zigzag"]
