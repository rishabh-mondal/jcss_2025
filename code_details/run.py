from os.path import join
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from time import time
from joblib import Parallel, delayed
from tqdm import tqdm


def ultralytics_to_geojson(
    task,
    classes,
    conf_threshold,
    prediction_dir,
    metadata_path,
    image_dir,
    target_geojson_path,
    save_dir,
    nms_iou,
):
    overall_init = time()
    init = time()
    print(f"Image directory: {image_dir}")
    print(f"Prediction directory: {prediction_dir}")

    image_paths = glob(join(image_dir, "*"))
    print(f"Number of images: {len(image_paths)}")
    prediction_path = join(prediction_dir, "labels", "*")
    print("Reading predictions from", prediction_path)
    prediction_paths = glob(prediction_path)
    print(f"Number of images with at least one prediction: {len(prediction_paths)}")
    metadata_gdf = gpd.read_file(metadata_path)
    print(f"Number of metadata entries: {len(metadata_gdf)}")
    target_gdf = gpd.read_file(target_geojson_path)
    print(f"Number of ground truth labels: {len(target_gdf)}")
    print(f"Time taken to read files: {time() - init:.2f}s")

    # Load predictions
    print("Predictions head")
    print(metadata_gdf.head(2))

    def obb_load_prediction(row):
        try:
            prediction = np.loadtxt(join(prediction_dir, "labels", f"{row['x']}_{row['y']}.txt"), ndmin=2)
        except FileNotFoundError:
            prediction = np.zeros((0, 10)) if task == "obb" else np.zeros((0, 6))

        # Preserve original prediction for later
        original_prediction = prediction.copy().tolist()
        original_prediction = ["_".join(map(str, x)) for x in original_prediction]

        # scale predictions
        min_x, min_y, max_x, max_y = row["geometry"].bounds
        prediction[:, 1:-1:2] = prediction[:, 1:-1:2] * (max_x - min_x) + min_x
        prediction[:, 2:-1:2] = (1 - prediction[:, 2:-1:2]) * (max_y - min_y) + min_y
        class_names = [classes[int(cls_id)] for cls_id in prediction[:, 0]]
        confidence = prediction[:, -1].tolist()

        box = prediction[:, 1:-1]
        return box, class_names, confidence, original_prediction

    init = time()
    metadata_gdf[["box", "class_name", "confidence", "yolo_label"]] = metadata_gdf.apply(
        obb_load_prediction, axis=1, result_type="expand"
    )
    print("Predictions head after getting box, class_name, confidence")
    print(metadata_gdf.head(2))
    print(f"Time taken to load predictions: {time() - init:.2f}s")

    init = time()
    print("Length before explode: ", len(metadata_gdf))
    metadata_gdf = metadata_gdf.apply(pd.Series.explode).reset_index(drop=True)
    print("Length after explode: ", len(metadata_gdf))
    metadata_gdf = metadata_gdf.dropna(subset=["box"]).reset_index(drop=True)
    print("Length after dropping NaN: ", len(metadata_gdf))
    print(metadata_gdf.head(2))
    print(f"Time taken to explode predictions: {time() - init:.2f}s")

    print("Length before conf filtering: ", len(metadata_gdf))
    metadata_gdf = metadata_gdf[metadata_gdf["confidence"] >= float(conf_threshold)]
    print("Length after conf filtering: ", len(metadata_gdf))

    init = time()
    if task == "obb":
        metadata_gdf["label_geometry"] = metadata_gdf["box"].apply(lambda x: Polygon(x.reshape(-1, 2)))
    print(metadata_gdf.head(2))
    print(f"Time taken to convert predictions to geometry: {time() - init:.2f}s")

    crs = metadata_gdf.crs
    metadata_gdf.drop(columns=["box", "x_idx", "y_idx", "geometry"], inplace=True)
    metadata_gdf.rename(columns={"label_geometry": "geometry"}, inplace=True)
    metadata_gdf.set_geometry("geometry", inplace=True)
    metadata_gdf.crs = crs

    ############# Overlap removal
    metadata_gdf.reset_index(drop=True, inplace=True)
    print(f"{type(metadata_gdf)=}")
    print(f"{metadata_gdf.columns=}")
    print(f"{metadata_gdf.crs=}")
    intersection_gdf = gpd.sjoin(metadata_gdf, metadata_gdf, predicate="intersects")
    # remove same points and duplicate pairs
    intersection_gdf = intersection_gdf[intersection_gdf.index < intersection_gdf.index_right][
        ["index_right"]
    ].reset_index(drop=False)
    intersection_gdf.rename(columns={"index": "index_left"}, inplace=True)

    def get_iou(row):
        geometry_left = metadata_gdf.loc[row.index_left, "geometry"]
        geometry_right = metadata_gdf.loc[row.index_right, "geometry"]
        return geometry_left.intersection(geometry_right).area / geometry_left.union(geometry_right).area

    intersection_gdf["iou"] = intersection_gdf.apply(get_iou, axis=1)

    def get_remove_indices(row):
        if row.iou >= nms_iou:
            left_area = metadata_gdf.loc[row.index_left, "geometry"].area
            right_area = metadata_gdf.loc[row.index_right, "geometry"].area
            return row.index_left if left_area > right_area else row.index_right

    intersection_gdf["index_remove"] = intersection_gdf.apply(get_remove_indices, axis=1)
    intersection_gdf.dropna(subset=["index_remove"], inplace=True)
    print("Size before NMS: ", len(metadata_gdf))
    metadata_gdf.drop(index=intersection_gdf["index_remove"], inplace=True)
    print("Size after NMS: ", len(metadata_gdf))

    print(f"{metadata_gdf.class_name.isnull().sum()=}")

    metadata_gdf["confidence"] = metadata_gdf["confidence"].astype(float)
    print(f"{metadata_gdf.dtypes=}")

    init = time()
    metadata_gdf.to_file(
        join(save_dir, f"predictions_{conf_threshold}.geojson"),
        driver="GeoJSON",
    )
    print(f"Time taken to save predictions: {time() - init:.2f}s")
    print(f"Total time taken: {time() - overall_init:.2f}s")
