import sqlite3
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file, url_for


app = Flask(__name__)
DB_PATH = Path(__file__).resolve().parent / "data" / "incidencias.db"
IMAGE_DIR = DB_PATH.parent / "incidencias_img"
PRIORITIES = {
    "Crítica": {"rank": 0, "color": "#dc2626"},
    "Alta": {"rank": 1, "color": "#f97316"},
    "Media": {"rank": 2, "color": "#eab308"},
    "Baja": {"rank": 3, "color": "#facc15"},
}
STATES = ("Pendiente", "En Licitación", "Reparado")
SORT_OPTIONS = {
    "fecha_desc": "Fecha: más recientes primero",
    "fecha_asc": "Fecha: más antiguos primero",
    "prioridad_desc": "Prioridad: mayor a menor",
    "prioridad_asc": "Prioridad: menor a mayor",
}


def get_priority(expediente):
    if (expediente["estado"] or "").casefold() == "reparado":
        priority = "Baja"
        score = 0
    else:
        potholes = expediente["baches"] or 0
        cracks = expediente["grietas"] or 0
        alligator_cracks = expediente["piel_cocodrilo"] or 0
        score = potholes * 3 + alligator_cracks * 2 + cracks
        if score >= 100:
            priority = "Crítica"
        elif score >= 50:
            priority = "Alta"
        elif score >= 15:
            priority = "Media"
        else:
            priority = "Baja"

    return {
        "prioridad": priority,
        "priority_score": score,
        **PRIORITIES[priority],
    }


def resolve_evidence_path(stored_path):
    if not stored_path:
        return None

    path = Path(stored_path)
    if path.is_absolute():
        candidate = path
    elif path.parts and path.parts[0].casefold() == IMAGE_DIR.name.casefold():
        candidate = DB_PATH.parent / path
    else:
        candidate = IMAGE_DIR / path

    try:
        candidate = candidate.resolve()
        candidate.relative_to(IMAGE_DIR.resolve())
    except ValueError:
        return None

    return candidate if candidate.is_file() else None


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
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
            """
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(expedientes)").fetchall()
        }
        if "imagen_original_path" not in columns:
            conn.execute(
                "ALTER TABLE expedientes ADD COLUMN imagen_original_path TEXT"
            )


@app.route("/")
def index():
    view = request.args.get("vista", "resumen")
    if view not in {"resumen", "inspecciones", "expedientes"}:
        view = "resumen"
    priority_filter = request.args.get("prioridad", "")
    if priority_filter not in PRIORITIES:
        priority_filter = ""
    state_filter = request.args.get("estado", "")
    if state_filter not in STATES:
        state_filter = ""
    sort_by = request.args.get("orden", "prioridad_desc")
    if sort_by not in SORT_OPTIONS:
        sort_by = "prioridad_desc"
    selected_id = request.args.get("id", "")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        summary = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(baches), 0) AS baches,
                COALESCE(SUM(grietas), 0) AS grietas,
                COALESCE(SUM(piel_cocodrilo), 0) AS piel_cocodrilo,
                COALESCE(SUM(CASE WHEN estado = 'Pendiente' THEN 1 ELSE 0 END), 0)
                    AS pendientes
            FROM expedientes
            """
        ).fetchone()
        incidencias = conn.execute(
            """
            SELECT id, fecha, ubicacion, total_defectos, estado
            FROM expedientes
            ORDER BY fecha DESC
            LIMIT 6
            """
        ).fetchall()
        expedientes = conn.execute(
            """
            SELECT id, fecha, ubicacion, lat, lon, baches, grietas,
                   piel_cocodrilo, total_defectos, estado, imagen_path,
                   imagen_original_path
            FROM expedientes
            """
        ).fetchall()

    records = []
    selected_expediente = None
    for expediente in expedientes:
        record = dict(expediente)
        record.update(get_priority(record))
        records.append(record)
        if record["id"] == selected_id:
            selected_expediente = record

    records.sort(key=lambda record: record["fecha"] or "", reverse=True)
    if view == "expedientes":
        if sort_by == "fecha_asc":
            records.reverse()
        elif sort_by == "prioridad_desc":
            records.sort(key=lambda record: record["rank"])
        elif sort_by == "prioridad_asc":
            records.sort(key=lambda record: record["rank"], reverse=True)
        if priority_filter:
            records = [
                record
                for record in records
                if record["prioridad"] == priority_filter
            ]
        if state_filter:
            records = [
                record
                for record in records
                if record["estado"] == state_filter
            ]

    selected_expediente = next(
        (record for record in records if record["id"] == selected_id),
        None,
    )

    selected_image = None
    selected_image_is_original = False
    processed_image = None
    if view == "expedientes" and selected_expediente:
        original_path = selected_expediente["imagen_original_path"]
        has_original_image = bool(resolve_evidence_path(original_path))
        evidence_path = (
            original_path if has_original_image else selected_expediente["imagen_path"]
        )
        if resolve_evidence_path(evidence_path):
            selected_image = url_for(
                "expediente_image", expediente_id=selected_id, tipo="original"
            )
            selected_image_is_original = has_original_image
        if original_path and selected_expediente["imagen_path"]:
            if resolve_evidence_path(selected_expediente["imagen_path"]):
                processed_image = url_for(
                    "expediente_image",
                    expediente_id=selected_id,
                    tipo="procesada",
                )

    map_points = [
        record
        for record in records
        if record["lat"] is not None and record["lon"] is not None
    ]

    return render_template(
        "index.html",
        view=view,
        summary=summary,
        incidencias=incidencias,
        expedientes=records,
        map_points=map_points,
        priority_filter=priority_filter,
        state_filter=state_filter,
        sort_by=sort_by,
        priorities=PRIORITIES,
        states=STATES,
        sort_options=SORT_OPTIONS,
        selected_expediente=selected_expediente,
        selected_image=selected_image,
        selected_image_is_original=selected_image_is_original,
        processed_image=processed_image,
    )


@app.route("/expedientes/<expediente_id>/imagen/<tipo>")
def expediente_image(expediente_id, tipo):
    if tipo not in {"original", "procesada"}:
        abort(404)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        expediente = conn.execute(
            """
            SELECT imagen_path, imagen_original_path
            FROM expedientes WHERE id = ?
            """,
            (expediente_id,),
        ).fetchone()

    if expediente is None:
        abort(404)

    stored_paths = (
        (
            expediente["imagen_original_path"],
            expediente["imagen_path"],
        )
        if tipo == "original"
        else (expediente["imagen_path"],)
    )
    image_path = None
    for stored_path in stored_paths:
        image_path = resolve_evidence_path(stored_path)
        if image_path is not None:
            break
    if image_path is None:
        abort(404)

    return send_file(image_path, conditional=True)


init_db()


if __name__ == "__main__":
    app.run(debug=True)
