# App de monitoreo (Evidently UI)
Dashboard con dos proyectos: **datos reales** (steps 401-743) y **escenario SIMULADO** de fraudes adaptativos.

## Ejecutar
Con el entorno conda activo, desde la raíz del repositorio:
    python monitoring_app/iniciar_app.py
Abre http://localhost:8000, elige el proyecto en la barra lateral y el periodo con el selector de fechas
(1 step = 1 hora simulada; el step 1 es el 1 de enero de 2026, 00:00).

## Qué muestra
- Proporción de variables con drift por ventana (data drift).
- Precision, recall y F1 por ventana (model drift).
- Tasa de fraude real (prior drift) y volumen de transacciones.
- En cada ventana, el reporte completo de Evidently.

El workspace se regenera ejecutando las celdas 7 y 8 de `notebooks/04_monitoreo.ipynb`.