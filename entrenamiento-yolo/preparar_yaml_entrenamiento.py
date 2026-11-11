import argparse
from pathlib import Path

import yaml

from dataset_utils import discover_splits, load_dataset


EXPECTED_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
]
DEFAULT_SOURCE = Path(r"D:\Entrenamiento\RDD2022\data\data.yaml")
DEFAULT_OUTPUT = Path(r"D:\Entrenamiento\RDD2022\data\asfalto_rdd2022.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea una copia local del YAML con ruta absoluta, sin modificar "
            "el archivo descargado."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, root, names = load_dataset(args.source)
    actual_names = [names[index] for index in sorted(names)]
    if (
        sorted(names) != list(range(4))
        or actual_names != EXPECTED_NAMES
        or config.get("nc") != 4
    ):
        raise ValueError(
            f"El YAML no tiene las cuatro clases esperadas en orden. "
            f"Esperado: {EXPECTED_NAMES}; actual: {actual_names}; "
            f"nc={config.get('nc')}"
        )

    config["path"] = str(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

    prepared_config, prepared_root, prepared_names = load_dataset(args.output)
    splits = discover_splits(args.output, prepared_config, prepared_root)
    for split_name in ("train", "val", "test"):
        split = splits.get(split_name)
        if split is None or not split["images"]:
            raise FileNotFoundError(
                f"No se encontraron imágenes para {split_name}; "
                f"revisa las rutas del YAML descargado."
            )
        if not split["label_roots"] or not any(
            path.is_dir() for path in split["label_roots"]
        ):
            raise FileNotFoundError(
                f"No se encontraron etiquetas para {split_name}."
            )

    print(f"YAML original conservado: {args.source.resolve()}")
    print(f"YAML preparado: {args.output.resolve()}")
    print(f"Raíz absoluta: {prepared_root}")
    print(f"Clases: {[prepared_names[index] for index in sorted(prepared_names)]}")
    for split_name, split in splits.items():
        print(
            f"{split_name}: {len(split['images'])} imágenes; "
            f"rutas de etiqueta: {split['label_roots']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
