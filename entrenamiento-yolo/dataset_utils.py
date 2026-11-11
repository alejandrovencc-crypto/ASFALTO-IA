from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_dataset(data_yaml: Path) -> tuple[dict[str, Any], Path, dict[int, str]]:
    data_yaml = data_yaml.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"No se encuentra el YAML del dataset: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"El YAML no contiene un objeto de configuración: {data_yaml}")

    configured_root = config.get("path", ".")
    root = Path(configured_root)
    if not root.is_absolute():
        root = data_yaml.parent / root
    root = root.resolve()

    raw_names = config.get("names", {})
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): str(name) for index, name in raw_names.items()}
    else:
        raise ValueError("El campo 'names' debe ser una lista o un diccionario.")
    if not names:
        raise ValueError("El YAML no define nombres de clases.")

    return config, root, names


def _resolve_path(value: str | Path, root: Path, yaml_parent: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    rooted = (root / path).resolve()
    if rooted.exists():
        return rooted
    return (yaml_parent / path).resolve()


def discover_splits(
    data_yaml: Path, config: dict[str, Any], root: Path
) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    yaml_parent = data_yaml.expanduser().resolve().parent
    split_keys = {"train": "train", "val": "val", "test": "test"}
    if "val" not in config and "valid" in config:
        split_keys["val"] = "valid"

    for split_name, config_key in split_keys.items():
        value = config.get(config_key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        image_paths = [
            _resolve_path(item, root, yaml_parent) for item in values if item
        ]
        images: list[Path] = []
        for image_path in image_paths:
            if image_path.is_dir():
                images.extend(
                    path
                    for path in image_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
            elif image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image_path)
            elif image_path.is_file() and image_path.suffix.lower() == ".txt":
                for line in image_path.read_text(encoding="utf-8-sig").splitlines():
                    item = line.strip()
                    if item:
                        listed_path = Path(item)
                        images.append(
                            listed_path.resolve()
                            if listed_path.is_absolute()
                            else (root / listed_path).resolve()
                        )

        label_roots: list[Path] = []
        for image_path in image_paths:
            split_directory_names = {split_name.lower(), str(config_key).lower()}
            if split_name == "val":
                split_directory_names.add("valid")
            if (
                image_path.is_dir()
                and image_path.name.lower() in split_directory_names
            ):
                if image_path.parent.name.lower() == "images":
                    label_root = image_path.parent.parent / "labels" / image_path.name
                else:
                    label_root = image_path.parent / "labels" / image_path.name
                label_roots.append(label_root.resolve())
            elif image_path.is_file():
                label_roots.append(
                    (image_path.parent.parent / "labels" / split_name).resolve()
                )

        splits[split_name] = {
            "config_key": config_key,
            "image_paths": image_paths,
            "images": sorted(set(images)),
            "label_roots": sorted(set(label_roots)),
        }
    return splits


def label_path_for_image(
    image_path: Path, image_dir: Path, label_root: Path
) -> Path:
    try:
        relative = image_path.relative_to(image_dir)
    except ValueError:
        relative = Path(image_path.name)
    return (label_root / relative).with_suffix(".txt")


def read_yolo_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    with label_path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_value = float(fields[0])
                coordinates = tuple(float(value) for value in fields[1:])
            except ValueError:
                continue
            if not class_value.is_integer():
                continue
            rows.append((int(class_value), *coordinates))
    return rows
