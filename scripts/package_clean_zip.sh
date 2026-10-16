#!/usr/bin/env bash
# ==============================================================================
# JORO-HPC — Generador de Archivo ZIP Limpio para Entrega y Respaldo
# Excluye directorios build/, data/, y metadatos internos de git (.git/)
# ==============================================================================
set -euo pipefail

OUTPUT_FILE="${1:-joro-hpc-clean.zip}"

echo "Generando paquete limpio de JORO-HPC desde el commit HEAD..."
git archive --format=zip --output="${OUTPUT_FILE}" HEAD

ZIP_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
echo "✔ Paquete generado exitosamente: ${OUTPUT_FILE} (${ZIP_SIZE})"
echo "Contenido verificado sin artefactos de compilación ni bases de datos locales."
