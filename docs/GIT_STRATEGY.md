# Estrategia de git

Se usa una variante simplificada de **GitHub Flow** con una rama de integración.

## Ramas

| Rama | Propósito |
|---|---|
| `main` | Siempre estable. Solo recibe cambios mediante Pull Request y es la rama de la versión publicada. |
| `development` | Rama de trabajo. Todos los commits del proyecto se hacen aquí. |

## Flujo

1. El repositorio se creó en GitHub con README, `.gitignore` de Python y licencia MIT (primer commit en `main`).
2. Se creó `development` desde `main` y se trabajó ahí.
3. Cada avance se subió con un commit descriptivo (`git add`, `git commit`, `git push`).
4. Al terminar, se abrió una **Pull Request de `development` hacia `main`** describiendo los cambios, y se fusionó con **merge** (sin squash, para conservar el historial).
5. Sobre `main` se creó el **release 1.0.0** con sus notas.

## Convención de commits

Prefijos en español, en el estilo *Conventional Commits*:

- `feat:` nueva funcionalidad (notebooks, app)
- `docs:` documentación
- `chore:` estructura, configuración, `.gitignore`
- `fix:` correcciones

Ejemplos: `chore: estructura inicial del proyecto y gitignore`, `feat: notebook 2 con micro-GA, MLflow y modelo registrado`.

## Qué no se versiona (`.gitignore`)

- Datos pesados: `data/raw/*.csv`, `data/processed/*.csv` y `data/processed/full/`.
- Artefactos locales de MLflow: `mlruns/`, `mlartifacts/`, `*.pkl`, `*.joblib` (el modelo vive en el Model Registry de DagsHub).
- Secretos: `.env` (credenciales de DagsHub).
- Basura de Python, Jupyter y editor.

Se usan archivos `.gitkeep` para conservar carpetas vacías.

## Versionado

- Versión semántica. Este proyecto se publica como **1.0.0**.
- Para la fecha de entrega no debe haber commits posteriores al release.

## Reproducibilidad

Los experimentos, métricas, parámetros y artefactos están en MLflow (DagsHub). El modelo productivo se carga por su alias (`models:/fraude-paysim-lgbm@produccion`), no por archivos locales.