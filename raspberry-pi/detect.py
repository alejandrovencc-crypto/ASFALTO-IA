import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Detecta daños en una imagen usando YOLOv8."
    )
    parser.add_argument("image", type=Path, help="Ruta de la imagen que se analizará.")
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "yolov8n.pt",
        help="Ruta a los pesos YOLOv8 (por defecto: yolov8n.pt junto al script).",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results",
        help="Carpeta donde se guardará la imagen anotada.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"No se encuentra la imagen: {args.image}")
    if not args.model.is_file():
        parser.error(f"No se encuentran los pesos del modelo: {args.model}")
    if not 0 < args.confidence <= 1:
        parser.error("--confidence debe estar entre 0 y 1.")
    if args.image_size <= 0:
        parser.error("--image-size debe ser mayor que 0.")

    return args


def main():
    args = parse_args()
    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.image),
        conf=args.confidence,
        imgsz=args.image_size,
        project=str(args.output.parent),
        name=args.output.name,
        exist_ok=True,
        save=True,
    )

    detections = sum(len(result.boxes) for result in results)
    print(f"Detecciones: {detections}")
    print(f"Imagen anotada guardada en: {args.output}")


if __name__ == "__main__":
    main()
