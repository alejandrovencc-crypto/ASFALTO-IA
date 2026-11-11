import argparse
import math
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from dataset_utils import (
    discover_splits,
    label_path_for_image,
    load_dataset,
    read_yolo_rows,
)


DEFAULT_YAML = Path(r"D:\Entrenamiento\RDD2022\data\asfalto_rdd2022.yaml")
DEFAULT_OUTPUT = Path(r"D:\Entrenamiento\resultados\muestra_anotaciones")
COLORS = [(40, 190, 80), (255, 170, 35), (220, 55, 55), (40, 140, 255)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera muestras anotadas y candidatos para revisión humana."
    )
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_class <= 0:
        raise ValueError("--per-class debe ser mayor que cero.")
    config, root, names = load_dataset(args.data_yaml)
    splits = discover_splits(args.data_yaml, config, root)
    rng = random.Random(args.seed)
    selected: dict[Path, tuple[str, set[int]]] = {}
    class_images: dict[int, list[tuple[Path, Path, Path, str]]] = defaultdict(list)
    pothole_candidates: list[str] = []

    for split_name, split in splits.items():
        for image_path in split["images"]:
            for image_dir in split["image_paths"]:
                if not image_dir.is_dir():
                    continue
                for label_root in split["label_roots"]:
                    label_path = label_path_for_image(image_path, image_dir, label_root)
                    if not label_path.is_file():
                        continue
                    rows = read_yolo_rows(label_path)
                    with Image.open(image_path) as source:
                        image_width, image_height = source.size
                    image_classes = {
                        class_id
                        for class_id, x, y, width, height in rows
                        if class_id in names
                        and all(
                            math.isfinite(value)
                            for value in (x, y, width, height)
                        )
                        and width > 0
                        and height > 0
                    }
                    for class_id in image_classes:
                        class_images[class_id].append(
                            (image_path, label_path, image_dir, split_name)
                        )
                    for class_id, _, _, width, height in rows:
                        if (
                            class_id == 3
                            and math.isfinite(width)
                            and math.isfinite(height)
                            and width > 0
                            and height > 0
                        ):
                            pixel_width = width * image_width
                            pixel_height = height * image_height
                            ratio = max(
                                pixel_width / pixel_height,
                                pixel_height / pixel_width,
                            )
                            if ratio >= 3:
                                pothole_candidates.append(
                                    f"- {image_path} | etiqueta {label_path} | "
                                    f"aspecto={ratio:.2f} | caja muy alargada: "
                                    "revisar visualmente posible marca o línea."
                                )
                    break

    for class_id in sorted(names):
        candidates = class_images.get(class_id, [])
        rng.shuffle(candidates)
        for image_path, _, _, split_name in candidates[: args.per_class]:
            if image_path in selected:
                selected[image_path][1].add(class_id)
            else:
                selected[image_path] = (split_name, {class_id})

    if not selected:
        raise RuntimeError(
            "No se encontraron imágenes etiquetadas para visualizar. "
            "Comprueba la auditoría y las rutas del YAML."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    report = [
        "REVISIÓN VISUAL DE ANOTACIONES — REQUIERE VALIDACIÓN HUMANA",
        f"YAML: {args.data_yaml.resolve()}",
        f"Semilla de muestreo: {args.seed}",
        f"Imágenes anotadas guardadas: {len(selected)}",
        "",
        "Muestras por imagen:",
    ]
    contact_tiles: list[Image.Image] = []
    font = ImageFont.load_default()

    for index, (image_path, (split_name, _)) in enumerate(
        sorted(selected.items(), key=lambda item: (item[1][0], str(item[0]))),
        start=1,
    ):
        split = splits[split_name]
        label_path = None
        image_dir = None
        for candidate_dir in split["image_paths"]:
            if not candidate_dir.is_dir():
                continue
            for label_root in split["label_roots"]:
                candidate_label = label_path_for_image(
                    image_path, candidate_dir, label_root
                )
                if candidate_label.is_file():
                    image_dir = candidate_dir
                    label_path = candidate_label
                    break
            if label_path:
                break
        if label_path is None or image_dir is None:
            continue

        rows = read_yolo_rows(label_path)
        with Image.open(image_path) as source:
            annotated = source.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        width_px, height_px = annotated.size
        present_classes = set()
        for class_id, x, y, box_width, box_height in rows:
            if (
                class_id not in names
                or not all(
                    math.isfinite(value)
                    for value in (x, y, box_width, box_height)
                )
                or box_width <= 0
                or box_height <= 0
            ):
                continue
            present_classes.add(class_id)
            left = (x - box_width / 2) * width_px
            top = (y - box_height / 2) * height_px
            right = (x + box_width / 2) * width_px
            bottom = (y + box_height / 2) * height_px
            color = COLORS[class_id % len(COLORS)]
            label = names[class_id]
            draw.rectangle((left, top, right, bottom), outline=color, width=3)
            text_box = draw.textbbox((left, top), label, font=font)
            draw.rectangle(text_box, fill=color)
            draw.text((left, top), label, fill="white", font=font)

        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem)
        output_name = f"{index:03d}_{split_name}_{safe_stem}.jpg"
        output_path = args.output / output_name
        annotated.save(output_path, quality=92)
        report.append(
            f"- {output_path.name}: partición={split_name}; "
            f"clases={[names[class_id] for class_id in sorted(present_classes)]}; "
            f"origen={image_path}"
        )
        tile = ImageOps.contain(annotated, (480, 360))
        contact_tiles.append(tile.copy())

    columns = 3
    tile_width, tile_height = 480, 360
    rows = (len(contact_tiles) + columns - 1) // columns
    contact_sheet = Image.new(
        "RGB", (columns * tile_width, rows * tile_height), "white"
    )
    for index, tile in enumerate(contact_tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        contact_sheet.paste(tile, (x, y))
    contact_path = args.output / "contact_sheet.jpg"
    contact_sheet.save(contact_path, quality=90)

    report.extend(
        [
            "",
            f"Hoja de contacto: {contact_path}",
            "",
            "CANDIDATOS HEURÍSTICOS PARA REVISAR (no son errores confirmados)",
            "Se listan cajas etiquetadas como pothole con relación de aspecto en "
            "píxeles >= 3. "
            "Una caja alargada puede ser una marca vial, pero también puede ser una "
            "anotación válida. Revise las imágenes originales y no cambie las "
            "etiquetas automáticamente.",
        ]
    )
    report.extend(pothole_candidates or ["- No se detectaron cajas de bache alargadas."])
    report.extend(
        [
            "",
            "En todas las muestras, compruebe especialmente: grietas longitudinales "
            "frente a transversales; patrón alligator; baches reales frente a líneas "
            "blancas, sombras, juntas y marcas del pavimento. La heurística no puede "
            "decidir la clase semántica y la revisión humana es obligatoria.",
        ]
    )
    report_path = args.output.parent / "revision_visual.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Imágenes anotadas y hoja de contacto: {args.output.resolve()}")
    print(f"Informe para revisión humana: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
