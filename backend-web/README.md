# Backend web

La interfaz Flask lee los expedientes guardados en `data/incidencias.db`.

Desde PowerShell, en la raíz del proyecto, instala las dependencias en el entorno de Python que usarás para iniciar la aplicación:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app backend-web/app.py run --debug
```

Si todavía no has creado `.venv`, puedes instalar y ejecutar con el Python seleccionado en VS Code:

```powershell
python -m pip install -r requirements.txt
python -m flask --app backend-web/app.py run --debug
```

En VS Code, selecciona `.venv\Scripts\python.exe` con **Python: Select Interpreter** para que coincida con ese entorno.

## Probar el modelo YOLOv8 con una imagen

El módulo de inspección usa los pesos locales `modelos/best.pt`; no
descarga otro modelo. Desde PowerShell, en la raíz del proyecto, instala las
dependencias e inicia Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run backend-web/streamlit_app.py
```

Abre <http://localhost:8501>, selecciona **Inspección por IA (YOLOv8)** en el
menú lateral, arrastra una imagen JPG o PNG y pulsa **Ejecutar Análisis de
Defectos**. Tras revisar el resultado, puedes indicar ubicación y coordenadas y
pulsar **Registrar Expediente en Base de Datos** para guardarlo en el histórico.
El archivo del modelo debe estar en `modelos/best.pt`.

Como alternativa, el detector de línea de comandos acepta la ruta del modelo:

```powershell
python raspberry-pi/detect.py .\ruta\a\foto.jpg --model .\modelos\best_calzada.pt
```

La imagen anotada se guarda en `raspberry-pi/results/` de forma predeterminada.

La prioridad se calcula automáticamente: baches ×3, piel de cocodrilo ×2 y grietas ×1. Una puntuación de 100 o más es crítica, 50 o más alta, 15 o más media y el resto baja. Los expedientes reparados quedan en prioridad baja. Al registrar nuevos análisis se conserva tanto la fotografía original como la imagen procesada.

En la pestaña **Expedientes** puedes ordenar por fecha o prioridad y combinar filtros por estado y prioridad. La selección se conserva al abrir o cerrar el detalle de un expediente.

La interfaz web abre por defecto en modo oscuro. El botón del encabezado alterna entre modo oscuro y claro y guarda la última elección en el navegador (`localStorage`), también al recargar o cambiar de sección.
