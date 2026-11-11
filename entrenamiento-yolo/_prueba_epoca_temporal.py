import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import entrenar_yolov8 as training
from ultralytics import YOLO


def main() -> None:
    training.validate_dataset()
    model = YOLO(str(training.INITIAL_MODEL))
    model.train(
        data=str(training.DATA_YAML),
        epochs=1,
        imgsz=640,
        batch=4,
        workers=2,
        device=0,
        amp=True,
        cache=False,
        project=str(training.OUTPUT_PROJECT),
        name="yolov8n_rdd2022_epoca_prueba_2",
        exist_ok=False,
        pretrained=True,
        patience=20,
        save=True,
    )
    run_dir = training.OUTPUT_PROJECT / "yolov8n_rdd2022_epoca_prueba_2"
    print("PRUEBA_EPOCA_COMPLETADA")
    print(f"RUN_DIR: {run_dir}")
    print(f"BEST: {run_dir / 'weights' / 'best.pt'}")
    print(f"LAST: {run_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
