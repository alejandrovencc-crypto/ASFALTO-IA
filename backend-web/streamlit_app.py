import datetime
import json
import os
import sqlite3
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_folium import st_folium
from ultralytics import YOLO

# -------------------------------------------------------------------
# CONFIGURACIÓN Y BASE DE DATOS LOCAL (SQLite)
# -------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_NAME = str(DATA_DIR / "incidencias.db")
IMG_DIR = str(DATA_DIR / "incidencias_img")

os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(IMG_DIR):
  os.makedirs(IMG_DIR)


def init_db():
  conn = sqlite3.connect(DB_NAME)
  c = conn.cursor()
  c.execute("""
        CREATE TABLE IF NOT EXISTS expedientes (
            id TEXT PRIMARY KEY,
            fecha TEXT,
            ubicacion TEXT,
            lat REAL,
            lon REAL,
            baches INTEGER,
            grietas INTEGER,
            piel_cocodrilo INTEGER,
            total_defectos INTEGER,
            estado TEXT,
            imagen_path TEXT,
            imagen_original_path TEXT
        )
    """)
  columns = {row[1] for row in c.execute("PRAGMA table_info(expedientes)")}
  if "imagen_original_path" not in columns:
    c.execute("ALTER TABLE expedientes ADD COLUMN imagen_original_path TEXT")
  conn.commit()
  conn.close()


init_db()

# -------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Gestión Vial - San Javier (B2G)",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e9ecef; }
        div.stButton > button:first-child { background-color: #0d6efd; color: white; font-weight: 600; border-radius: 8px; width: 100%; padding: 0.6rem; border: none; }
        div.stButton > button:first-child:hover { background-color: #0b5ed7; }
        .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# BARRA LATERAL
# -------------------------------------------------------------------
with st.sidebar:
  st.image("https://img.icons8.com/color/96/road.png", width=70)
  st.title("Ayuntamiento\nde San Javier")
  st.caption("Concejalía de Servicios Públicos e Infraestructuras")
  st.markdown("---")

  modo = st.radio(
      "📂 Módulos del Sistema",
      [
          "🗺️ Mapa de Expedientes (GIS)",
          "🤖 Inspección por IA (YOLOv8)",
          "🗄️ Histórico de Expedientes",
      ],
  )

  st.markdown("---")
  st.info(
      "**Estado del Sistema:**\n- Base de Datos: SQLite (`incidencias.db`)\n-"
      " Edge Device: Luckfox Pico\n- Modelo: RDD2022 Multiclase"
  )

# -------------------------------------------------------------------
# MÓDULO 1: MAPA INTERACTIVO GIS
# -------------------------------------------------------------------
if modo == "🗺️ Mapa de Expedientes (GIS)":
  st.title("📍 Visor Cartográfico de Incidencias Viarias")
  st.markdown(
      "Geolocalización de daños estructurales registrados en el término"
      " municipal de San Javier."
  )

  conn = sqlite3.connect(DB_NAME)
  df_exp = pd.read_sql_query("SELECT * FROM expedientes", conn)
  conn.close()

  col_m1, col_m2, col_m3, col_m4 = st.columns(4)
  with col_m1:
    st.metric("Total Expedientes", len(df_exp))
  with col_m2:
    st.metric(
        "Baches Críticos (D40)",
        int(df_exp["baches"].sum()) if not df_exp.empty else 0,
    )
  with col_m3:
    st.metric(
        "Piel de Cocodrilo (D20)",
        int(df_exp["piel_cocodrilo"].sum()) if not df_exp.empty else 0,
    )
  with col_m4:
    pendientes = (
        len(df_exp[df_exp["estado"] == "Pendiente"]) if not df_exp.empty else 0
    )
    st.metric("Pendientes Licitación", pendientes)

  st.markdown("<br>", unsafe_allow_html=True)

  # Coordenadas centro San Javier
  lat_san_javier, lon_san_javier = 37.8063, -0.8374

  # Mapa base OpenStreetMap
  m = folium.Map(
      location=[lat_san_javier, lon_san_javier],
      zoom_start=14,
      tiles="OpenStreetMap",
  )

  # Dibujar expedientes almacenados en la BD
  if not df_exp.empty:
    for _, row in df_exp.iterrows():
      if row["baches"] > 0:
        color_hex = "#d9534f"  # Rojo para Baches (D40)
      elif row["piel_cocodrilo"] > 0:
        color_hex = "#f0ad4e"  # Naranja para Piel de Cocodrilo (D20)
      else:
        color_hex = "#0275d8"  # Azul para Grietas (D00/D10)

      popup_html = f"""
            <div style='font-family: Arial, sans-serif; width: 200px;'>
                <h4 style='margin:0; color:#0d6efd;'>{row['id']}</h4>
                <p style='margin:5px 0;'><b>Ubicación:</b> {row['ubicacion']}</p>
                <p style='margin:5px 0;'><b>Fecha:</b> {row['fecha']}</p>
                <p style='margin:5px 0;'><b>Total Defectos:</b> {row['total_defectos']}</p>
                <p style='margin:5px 0;'><b>Estado:</b> <span style='color:{color_hex}; font-weight:bold;'>{row['estado']}</span></p>
            </div>
            """

      folium.CircleMarker(
          location=[row["lat"], row["lon"]],
          radius=9,
          popup=folium.Popup(popup_html, max_width=250),
          tooltip=f"{row['id']}: {row['ubicacion']}",
          color=color_hex,
          fill=True,
          fill_color=color_hex,
          fill_opacity=0.8,
      ).add_to(m)

  st_folium(m, width="100%", height=520, key="mapa_san_javier")

# -------------------------------------------------------------------
# MÓDULO 2: INSPECCIÓN POR IA Y REGISTRO
# -------------------------------------------------------------------
elif modo == "🤖 Inspección por IA (YOLOv8)":
  st.title("🤖 Módulo de Diagnóstico Automático por Visión Artificial")
  st.markdown(
      "Análisis automatizado de imágenes de calzada mediante redes neuronales"
      " YOLOv8 y taxonomía internacional **RDD2022**."
  )

  with st.expander(
      "⚙️ Parámetros avanzados de inferencia (Edge AI)", expanded=True
  ):
    col_c1, col_c2 = st.columns(2)
    with col_c1:
      conf_threshold = st.slider(
          "Umbral de Confianza (Sensibilidad de Detección)",
          min_value=0.05,
          max_value=0.50,
          value=0.12,
          step=0.01,
      )
    with col_c2:
      img_size = st.select_slider(
          "Resolución de Procesamiento (Píxeles)",
          options=[640, 800, 1024, 1280],
          value=1024,
      )

  uploaded_file = st.file_uploader(
      "📤 Seleccione o arrastre una fotografía de la vía pública (JPG, PNG)",
      type=["jpg", "jpeg", "png"],
  )

  if uploaded_file is not None:
    col_img1, col_img2 = st.columns(2)

    with col_img1:
      st.subheader("📷 Fotografía Original")
      image = Image.open(uploaded_file)
      st.image(image, use_container_width=True)

    with col_img2:
      st.subheader("📊 Diagnóstico IA en Directo")
      if st.button("🚀 Ejecutar Análisis de Defectos"):
        with st.spinner(
            "Procesando red neuronal y clasificando patologías..."
        ):
          weights_path = hf_hub_download(
              repo_id="ozair23/yolov8-road-damage-detector", filename="best.pt"
          )
          model = YOLO(weights_path)

          img_np = np.array(image)
          results = model(img_np, conf=conf_threshold, imgsz=img_size)

          potholes_count = 0
          cracks_count = 0
          alligator_count = 0
          total_detections = 0

          for r in results:
            for box in r.boxes:
              total_detections += 1
              cls_id = int(box.cls[0])
              class_name = model.names[cls_id].lower()

              if (
                  "pothole" in class_name
                  or "d40" in class_name
                  or "bache" in class_name
              ):
                potholes_count += 1
              elif (
                  "alligator" in class_name
                  or "d20" in class_name
                  or "cocodrilo" in class_name
              ):
                alligator_count += 1
              else:
                cracks_count += 1

            im_array = r.plot(
                line_width=2, font_size=1, conf=True, labels=True
            )

          st.session_state["im_array"] = im_array
          st.session_state["p_count"] = potholes_count
          st.session_state["c_count"] = cracks_count
          st.session_state["a_count"] = alligator_count
          st.session_state["t_count"] = total_detections
          st.session_state["analyzed"] = True

    if st.session_state.get("analyzed", False):
      with col_img2:
        st.image(st.session_state["im_array"], use_container_width=True)

      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("### 📋 Resumen Ejecutivo de Daños Detectados")

      m1, m2, m3, m4 = st.columns(4)
      with m1:
        st.metric(
            label="🕳️ Baches (D40)", value=st.session_state["p_count"]
        )
      with m2:
        st.metric(
            label="⚡ Grietas (D00/D10)", value=st.session_state["c_count"]
        )
      with m3:
        st.metric(
            label="🐊 Piel de Cocodrilo (D20)",
            value=st.session_state["a_count"],
        )
      with m4:
        st.metric(
            label="📊 Total Incidencias", value=st.session_state["t_count"]
        )

      st.markdown("---")
      st.markdown("### 💾 Guardar Expediente Municipal")

      col_form1, col_form2, col_form3 = st.columns([2, 1, 1])
      with col_form1:
        ubicacion_input = st.text_input(
            "Ubicación / Calle", "Av. Doctor Artero Guirao, San Javier"
        )
      with col_form2:
        lat_input = st.number_input("Latitud", value=37.8063, format="%.4f")
      with col_form3:
        lon_input = st.number_input("Longitud", value=-0.8374, format="%.4f")

      if st.button("📥 Registrar Expediente en Base de Datos"):
        exp_id = f"EXP-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        fecha_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        img_filename = str(Path(IMG_DIR) / f"{exp_id}.jpg")

        # Guardar imagen procesada
        processed_img = Image.fromarray(st.session_state["im_array"])
        processed_img.save(img_filename)
        original_img_filename = str(Path(IMG_DIR) / f"{exp_id}_original.jpg")
        image.convert("RGB").save(original_img_filename, format="JPEG", quality=90)

        # Insertar en SQLite
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
                    INSERT INTO expedientes (
                        id, fecha, ubicacion, lat, lon, baches, grietas,
                        piel_cocodrilo, total_defectos, estado, imagen_path,
                        imagen_original_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
            (
                exp_id,
                fecha_str,
                ubicacion_input,
                lat_input,
                lon_input,
                st.session_state["p_count"],
                st.session_state["c_count"],
                st.session_state["a_count"],
                st.session_state["t_count"],
                "Pendiente",
                img_filename,
                original_img_filename,
            ),
        )
        conn.commit()
        conn.close()

        st.success(
            f"✅ ¡Expediente {exp_id} registrado e insertado en la base de datos!"
        )
        st.balloons()

# -------------------------------------------------------------------
# MÓDULO 3: HISTÓRICO Y REGISTRO DE EXPEDIENTES
# -------------------------------------------------------------------
elif modo == "🗄️ Histórico de Expedientes":
  st.title("🗄️ Base de Datos e Histórico de Expedientes")
  st.markdown(
      "Consulta, filtrado y gestión de todas las inspecciones de calzada"
      " guardadas."
  )

  conn = sqlite3.connect(DB_NAME)
  df = pd.read_sql_query("SELECT * FROM expedientes ORDER BY fecha DESC", conn)
  conn.close()

  if df.empty:
    st.info(
        "Aún no hay expedientes guardados. Realiza un análisis en la pestaña"
        " 'Inspección por IA' y pulsa en 'Registrar Expediente'."
    )
  else:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
      filtro_estado = st.selectbox(
          "Filtrar por Estado", ["Todos", "Pendiente", "En Licitación", "Reparado"]
      )
    with col_f2:
      search_term = st.text_input("Buscar por Calle / ID", "")

    df_filtered = df.copy()
    if filtro_estado != "Todos":
      df_filtered = df_filtered[df_filtered["estado"] == filtro_estado]
    if search_term:
      df_filtered = df_filtered[
          df_filtered["ubicacion"].str.contains(search_term, case=False)
          | df_filtered["id"].str.contains(search_term, case=False)
      ]

    st.dataframe(
        df_filtered[
            [
                "id",
                "fecha",
                "ubicacion",
                "baches",
                "grietas",
                "piel_cocodrilo",
                "total_defectos",
                "estado",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 👁️ Detalle y Fotografía del Expediente")

    exp_seleccionado = st.selectbox(
        "Seleccione un Expediente para inspeccionar:", df_filtered["id"].tolist()
    )

    if exp_seleccionado:
      row = df_filtered[df_filtered["id"] == exp_seleccionado].iloc[0]

      col_det1, col_det2 = st.columns(2)
      with col_det1:
        st.write(f"**Código Expediente:** `{row['id']}`")
        st.write(f"**Fecha Inspección:** {row['fecha']}")
        st.write(f"**Ubicación:** {row['ubicacion']}")
        st.write(f"**Coordenadas GPS:** `{row['lat']:.4f}, {row['lon']:.4f}`")
        st.write(f"**Estado Actual:** `{row['estado']}`")

        nuevo_estado = st.selectbox(
            "Actualizar Estado del Expediente:",
            ["Pendiente", "En Licitación", "Reparado"],
            index=["Pendiente", "En Licitación", "Reparado"].index(
                row["estado"]
            ),
        )
        if st.button("💾 Actualizar Estado"):
          conn = sqlite3.connect(DB_NAME)
          c = conn.cursor()
          c.execute(
              "UPDATE expedientes SET estado = ? WHERE id = ?",
              (nuevo_estado, row["id"]),
          )
          conn.commit()
          conn.close()
          st.success("¡Estado actualizado con éxito!")
          st.rerun()

      with col_det2:
        image_path = Path(row["imagen_path"])
        if not image_path.is_absolute() and not image_path.is_file():
          image_path = DATA_DIR / image_path

        if image_path.is_file():
          st.image(
              Image.open(image_path),
              caption=f"Evidencia Fotográfica - {row['id']}",
              use_container_width=True,
          )
        else:
          st.warning("No se encontró la imagen asociada a este registro.")