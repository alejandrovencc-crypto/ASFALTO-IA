import argparse
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from dataset_utils import discover_splits, label_path_for_image, load_dataset


EXPECTED_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole",
]
BOX_EDGE_TOLERANCE = 1e-6
DEFAULT_YAML = Path(r"D:\Entrenamiento\RDD2022\data\asfalto_rdd2022.yaml")
DEFAULT_REPORT = Path(r"D:\Entrenamiento\resultados\auditoria_dataset.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita estructura, imágenes y etiquetas de un dataset YOLO."
    )
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, root, names = load_dataset(args.data_yaml)
    splits = discover_splits(args.data_yaml, config, root)
    ordered_names = [names[index] for index in sorted(names)]
    class_ids_are_contiguous = sorted(names) == list(range(len(names)))
    class_count = int(config.get("nc", len(names)))
    names_are_correct = (
        class_ids_are_contiguous
        and ordered_names == EXPECTED_NAMES
        and class_count == len(EXPECTED_NAMES)
    )
    problems: list[str] = []
    report: list[str] = [
        "AUDITORÍA DEL DATASET YOLO",
        f"YAML: {args.data_yaml.resolve()}",
        f"Raíz resuelta: {root}",
        f"Clases declaradas (nc={class_count}): {ordered_names}",
        f"Orden esperado: {EXPECTED_NAMES}",
        f"Orden de clases correcto: {names_are_correct}",
        "",
    ]

    if not names_are_correct:
        problems.append(
            "El YAML no define los IDs 0-3 y las cuatro clases esperadas "
            "en el orden solicitado."
        )
    if class_count != len(names):
        problems.append(
            f"nc={class_count}, pero el YAML declara {len(names)} nombres."
        )

    for split_name, split in splits.items():
        images: list[Path] = split["images"]
        label_roots: list[Path] = split["label_roots"]
        counts = Counter()
        empty_labels: list[Path] = []
        missing_labels: list[Path] = []
        corrupt_images: list[Path] = []
        bad_lines: list[str] = []
        label_files = [
            path
            for label_root in label_roots
            if label_root.is_dir()
            for path in label_root.rglob("*.txt")
        ]
        image_to_label: dict[Path, Path] = {}

        for image_path in images:
            label_candidates = [
                label_path_for_image(image_path, image_dir, label_root)
                for image_dir in split["image_paths"]
                if image_dir.is_dir()
                for label_root in label_roots
            ]
            label_path = next(
                (candidate for candidate in label_candidates if candidate.is_file()),
                None,
            )
            if label_path is None:
                missing_labels.append(image_path)
            else:
                image_to_label[image_path] = label_path
                label_text = label_path.read_text(encoding="utf-8-sig")
                if not label_text.strip():
                    empty_labels.append(label_path)
                for line_number, raw_line in enumerate(
                    label_text.splitlines(), start=1
                ):
                    fields = raw_line.split()
                    if not fields:
                        continue
                    if len(fields) != 5:
                        bad_lines.append(
                            f"{label_path}:{line_number}: se esperaban 5 campos, "
                            f"se encontraron {len(fields)}"
                        )
                        continue
                    try:
                        class_value = float(fields[0])
                        x, y, width, height = (float(value) for value in fields[1:])
                    except ValueError:
                        bad_lines.append(
                            f"{label_path}:{line_number}: clase o coordenada no numérica"
                        )
                        continue
                    if not class_value.is_integer():
                        bad_lines.append(
                            f"{label_path}:{line_number}: id de clase no entero "
                            f"({fields[0]})"
                        )
                    else:
                        class_id = int(class_value)
                        if class_id < 0 or class_id >= class_count:
                            bad_lines.append(
                                f"{label_path}:{line_number}: id de clase fuera de "
                                f"rango ({class_id})"
                            )
                        else:
                            counts[class_id] += 1
                    values = (x, y, width, height)
                    if not all(math.isfinite(value) for value in values):
                        bad_lines.append(
                            f"{label_path}:{line_number}: coordenadas no finitas"
                        )
                    elif (
                        any(value < 0 or value > 1 for value in values)
                        or width <= 0
                        or height <= 0
                        or x - width / 2 < -BOX_EDGE_TOLERANCE
                        or y - height / 2 < -BOX_EDGE_TOLERANCE
                        or x + width / 2 > 1 + BOX_EDGE_TOLERANCE
                        or y + height / 2 > 1 + BOX_EDGE_TOLERANCE
                    ):
                        bad_lines.append(
                            f"{label_path}:{line_number}: caja YOLO fuera del rango "
                            "normalizado [0,1] o con ancho/alto no positivo"
                        )

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, ValueError) as error:
                corrupt_images.append(image_path)
                problems.append(f"Imagen corrupta ({split_name}): {image_path}: {error}")

        paired_labels = set(image_to_label.values())
        orphan_labels = sorted(set(label_files) - paired_labels)
        report.extend(
            [
                f"PARTICIÓN {split_name}",
                f"  Directorios de imágenes: {split['image_paths']}",
                f"  Directorios de etiquetas: {label_roots}",
                f"  Imágenes: {len(images)}",
                f"  Archivos de etiquetas: {len(label_files)}",
                f"  Imágenes sin etiqueta: {len(missing_labels)}",
                f"  Etiquetas vacías: {len(empty_labels)}",
                f"  Etiquetas sin imagen correspondiente: {len(orphan_labels)}",
                f"  Imágenes corruptas: {len(corrupt_images)}",
                f"  Líneas de etiqueta inválidas: {len(bad_lines)}",
                "  Instancias por clase:",
            ]
        )
        if not images:
            problems.append(f"No se encontraron imágenes para {split_name}.")
        report.extend(
            f"    {class_id} ({names.get(class_id, 'CLASE_DESCONOCIDA')}): "
            f"{counts[class_id]}"
            for class_id in range(class_count)
        )
        for heading, paths in (
            ("Imágenes sin etiqueta", missing_labels),
            ("Etiquetas vacías", empty_labels),
            ("Etiquetas sin imagen", orphan_labels),
            ("Imágenes corruptas", corrupt_images),
        ):
            if paths:
                report.append(f"  Detalle: {heading}")
                report.extend(f"    {path}" for path in paths[:100])
                if len(paths) > 100:
                    report.append(f"    ... y {len(paths) - 100} elementos más")
        if bad_lines:
            report.append("  Detalle: líneas de etiqueta inválidas")
            report.extend(f"    {line}" for line in bad_lines[:200])
            if len(bad_lines) > 200:
                report.append(f"    ... y {len(bad_lines) - 200} líneas más")
        for count, description in (
            (len(missing_labels), "imágenes sin etiqueta"),
            (len(orphan_labels), "etiquetas sin imagen"),
            (len(bad_lines), "líneas de etiqueta inválidas"),
        ):
            if count:
                problems.append(f"{split_name}: {count} {description}.")
        if not label_roots or not any(path.is_dir() for path in label_roots):
            problems.append(f"No se encontró directorio de etiquetas para {split_name}.")
        report.append("")

    report.append("PROBLEMAS ESTRUCTURALES")
    report.extend(f"- {problem}" for problem in problems)
    if not problems:
        report.append("- No se detectaron problemas estructurales.")
    report.append(
        "\nNota: esta auditoría valida integridad y formato; no juzga semánticamente "
        "si una caja representa una grieta, un bache, una línea pintada o una sombra."
    )
    report.append(
        "Las etiquetas vacías se contabilizan, pero no se consideran un error: "
        "YOLO las interpreta como imágenes sin objetos anotados."
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Informe guardado en: {args.report.resolve()}")
    print(f"Clases esperadas en orden: {names_are_correct}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
