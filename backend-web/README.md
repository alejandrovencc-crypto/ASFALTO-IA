# Backend web

La interfaz Flask lee los expedientes guardados en `data/incidencias.db`.

Desde PowerShell, en la raíz del proyecto, instala las dependencias en el entorno de Python que usarás para iniciar la aplicación:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app backend-web/app.py run --debug
```

Si todavía no has creado `.venv`, puedes instalar y ejecutar con el Python seleccionado en VS Code:

```powershell
python -m pip install Flask
python -m flask --app backend-web/app.py run --debug
```

En VS Code, selecciona `.venv\Scripts\python.exe` con **Python: Select Interpreter** para que coincida con ese entorno.

La aplicación Streamlit existente se conserva en `streamlit_app.py`:

```powershell
streamlit run backend-web/streamlit_app.py
```

La prioridad se calcula automáticamente: baches ×3, piel de cocodrilo ×2 y grietas ×1. Una puntuación de 100 o más es crítica, 50 o más alta, 15 o más media y el resto baja. Los expedientes reparados quedan en prioridad baja. Al registrar nuevos análisis se conserva tanto la fotografía original como la imagen procesada.
