# Raspberry Pi / YOLOv8

Instala las dependencias del proyecto y ejecuta el detector con una imagen y el
modelo entrenado (desde la raíz del proyecto):

```powershell
python -m pip install -r requirements.txt
python raspberry-pi/detect.py .\ruta\a\imagen.jpg --model .\modelos\best_calzada.pt
```

La imagen anotada se guarda en `raspberry-pi/results/`. Si ejecutas el script
desde esta carpeta sin `--model`, seguirá usando `yolov8n.pt`.
