# Entrenamiento local YOLOv8 — ASFALTO-IA

Las herramientas usan el entorno existente `D:\Entrenamiento\venv`. No crean
otro entorno ni alteran las etiquetas originales. El entrenamiento no se inicia
hasta ejecutar explícitamente `entrenar_yolov8.py`.

## Activar el entorno

Desde PowerShell:

```powershell
& 'D:\Entrenamiento\venv\Scripts\Activate.ps1'
```

O ejecuta siempre el intérprete explícito:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' --version
```

En VS Code, el intérprete seleccionado debe ser
`D:\Entrenamiento\venv\Scripts\python.exe`.

## Comprobar la GPU

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA no disponible')"
```

El entorno auditado tiene PyTorch `2.6.0+cu126`, CUDA disponible y una GTX 1070
(compute capability 6.1). No actualices PyTorch sin revisar compatibilidad y
obtener autorización.

## Descargar RDD2022

El repositorio de Hugging Face es `dronefreak/RDD2022`. El dataset se guarda en
`D:\Entrenamiento\RDD2022`. Para reanudar o completar una descarga con
`huggingface_hub`:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='dronefreak/RDD2022', repo_type='dataset', local_dir=r'D:\Entrenamiento\RDD2022', max_workers=32))"
```

Comprueba el espacio disponible antes de una primera descarga. El YAML remoto
verificado está en `data\data.yaml`, con cuatro clases en este orden:

0. `longitudinal_crack`
1. `transverse_crack`
2. `alligator_crack`
3. `pothole`

El YAML descargado usa `path: .`, cuyo significado puede depender del directorio
de trabajo de Ultralytics. Conserva ese archivo original y prepara una copia con
ruta absoluta antes de auditar o visualizar:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' 'D:\ASFALTO-IA-main\ASFALTO-IA\entrenamiento-yolo\preparar_yaml_entrenamiento.py'
```

La copia queda en
`D:\Entrenamiento\RDD2022\data\asfalto_rdd2022.yaml`; comprueba que muestra las
tres particiones y sus directorios de etiquetas.

## Auditar el dataset

El script reutilizable está en `entrenamiento-yolo\auditar_dataset.py`. Genera
el informe en `D:\Entrenamiento\resultados\auditoria_dataset.txt`:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' 'D:\ASFALTO-IA-main\ASFALTO-IA\entrenamiento-yolo\auditar_dataset.py'
```

Se pueden especificar otras rutas:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' 'D:\ASFALTO-IA-main\ASFALTO-IA\entrenamiento-yolo\auditar_dataset.py' --data-yaml 'D:\Entrenamiento\RDD2022\data\asfalto_rdd2022.yaml' --report 'D:\Entrenamiento\resultados\auditoria_dataset.txt'
```

La auditoría cuenta imágenes, ficheros y objetos etiquetados por partición;
detecta imágenes sin etiqueta, etiquetas huérfanas, imágenes corruptas, IDs de
clase y cajas YOLO fuera de formato. Las etiquetas vacías se cuentan como
muestras negativas, que son válidas para YOLO.
Tolera una desviación de hasta `1e-6` en los bordes de las cajas para evitar
falsos positivos por redondeo decimal; no corrige ni modifica las etiquetas.

## Visualizar las anotaciones

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' 'D:\ASFALTO-IA-main\ASFALTO-IA\entrenamiento-yolo\visualizar_anotaciones.py'
```

Las imágenes anotadas se guardan en
`D:\Entrenamiento\resultados\muestra_anotaciones` y el resumen para revisión
humana en `D:\Entrenamiento\resultados\revision_visual.txt`. La hoja de contacto
es `contact_sheet.jpg`. Se marcan como candidatos —no como errores confirmados—
los baches con cajas muy alargadas. Revisa las líneas blancas, sombras, juntas y
marcas manualmente; el programa no cambia etiquetas.

## Configuración e inicio del entrenamiento

El script `entrenamiento-yolo\entrenar_yolov8.py` usa el peso YOLOv8n local
`raspberry-pi\yolov8n.pt`, comprueba el orden de clases y que existan las tres
particiones antes de entrenar. Configuración inicial:

- imagen: 640;
- batch: 4;
- workers: 2;
- dispositivo: GPU 0;
- AMP: activado;
- cache: desactivada para limitar uso de RAM.

No ejecutes este comando hasta revisar los informes de auditoría y anotaciones:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' 'D:\ASFALTO-IA-main\ASFALTO-IA\entrenamiento-yolo\entrenar_yolov8.py'
```

Los resultados se escribirán en
`D:\Entrenamiento\runs\yolov8n_rdd2022`. Los pesos estarán en
`weights\best.pt` y `weights\last.pt`. El script no permite sobrescribir una
ejecución existente con el mismo nombre.

## Inferencia

El detector CLI existente acepta un peso personalizado. Sustituye `foto.jpg`
por la imagen a evaluar:

```powershell
& 'D:\Entrenamiento\venv\Scripts\python.exe' 'D:\ASFALTO-IA-main\ASFALTO-IA\raspberry-pi\detect.py' 'D:\ruta\foto.jpg' --model 'D:\Entrenamiento\runs\yolov8n_rdd2022\weights\best.pt' --output 'D:\Entrenamiento\resultados\inferencia'
```

La imagen resultante se guardará en `D:\Entrenamiento\resultados\inferencia`.
