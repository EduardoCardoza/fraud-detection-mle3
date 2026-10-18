"""Inicia la app de monitoreo (Evidently UI) sobre el workspace de este proyecto."""
import sys
from pathlib import Path

from evidently.cli import app  # la misma función que ejecuta el comando `evidently`

RAIZ = Path(__file__).resolve().parent  # carpeta monitoring_app
PUERTO = "8000"

if __name__ == "__main__":
    print(f"Abre http://localhost:{PUERTO} en tu navegador (Ctrl+C para detener)")
    sys.argv = ["evidently", "ui", "--workspace", str(RAIZ / "workspace"), "--port", PUERTO]
    app()