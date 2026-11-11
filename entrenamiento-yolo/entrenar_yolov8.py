from pathlib import Path

from ultralytics import YOLO

from dataset_utils import (
    discover_splits,
    label_path_for_image,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = Path(r"D:\Entrenamiento\RDD2022\data\asfalto_rdd2022.yaml")
INITIAL_MODEL = PROJECT_ROOT / "raspberry-pi" / "yolov8n.pt"
OUTPUT_PROJECT = Path(r"D:\Entrenamiento\runs")
EXPECTED_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
]


def validate_dataset() -> None:
    config, root, names = load_dataset(DATA_YAML)
    actual_names = [names[index] for index in sorted(names)]
    if (
        sorted(names) != list(range(4))
        or actual_names != EXPECTED_NAMES
        or config.get("nc") != 4
    ):
        raise ValueError(
            f"Clases u orden incorrectos. Esperado: {EXPECTED_NAMES}; "
            f"actual: {actual_names}; nc={config.get('nc')}"
        )
    splits = discover_splits(DATA_YAML, config, root)
    for split_name in ("train", "val", "test"):
        split = splits.get(split_name)
        if split is None or not split["images"]:
            raise FileNotFoundError(
                f"No se encontraron imágenes para la partición {split_name}."
            )
        if not split["label_roots"] or not any(
            path.is_dir() for path in split["label_roots"]
        ):
            raise FileNotFoundError(
                f"No se encontraron etiquetas para la partición {split_name}."
            )
        missing_paths = [
            path for path in split["image_paths"] if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Rutas inexistentes en {split_name}: {missing_paths}"
            )
        missing_labels = []
        for image_path in split["images"]:
            label_candidates = [
                label_path_for_image(image_path, image_dir, label_root)
                for image_dir in split["image_paths"]
                if image_dir.is_dir()
                for label_root in split["label_roots"]
            ]
            if not any(path.is_file() for path in label_candidates):
                missing_labels.append(image_path)
                if len(missing_labels) == 10:
                    break
        if missing_labels:
            raise FileNotFoundError(
                f"Faltan etiquetas para imágenes de {split_name}: {missing_labels}"
            )
    if not INITIAL_MODEL.is_file():
        raise FileNotFoundError(f"No se encuentra el modelo inicial: {INITIAL_MODEL}")
    if not OUTPUT_PROJECT.is_dir():
        OUTPUT_PROJECT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    validate_dataset()
    model = YOLO(str(INITIAL_MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=4,
        workers=2,
        device=0,
        amp=True,
        cache=False,
        project=str(OUTPUT_PROJECT),
        name="yolov8n_rdd2022",
        exist_ok=False,
        pretrained=True,
        patience=20,
        save=True,
    )


if __name__ == "__main__":
    main()
