# Resultados

Archivos generados por `src.evaluate` y `src.figures`, incluidos tal cual en el artículo.

| Archivo | Contenido |
|---|---|
| `resultados.csv` | métricas por modelo (media y desviación entre semillas), parámetros, imágenes/s, horas de GPU |
| `por_par.csv.gz` | PSNR, SSIM y LPIPS de cada par de prueba, modelo y semilla, con el ángulo relativo |
| `angulos.csv` | PSNR medio por intervalo de ángulo relativo |
| `wilcoxon.csv` | pruebas de rangos con signo emparejadas, con corrección de Holm |
| `analisis.json` | cifras complementarias citadas en §IV–V: por semilla, por objeto, intervalos finos de ángulo, peores casos y convergencia |
| `costo.json` | horas de GPU totales, dispositivo y número de pares evaluados |
| `tabla_*.tex` | tablas en formato `booktabs` listas para `\input{}` |
| `fig_*.pdf` | figuras vectoriales dimensionadas para IEEEtran |
| `logs/` | copia de `log.json` y `summary.json` de cada run |
