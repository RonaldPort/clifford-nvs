# Síntesis de nuevas vistas con interacción de Clifford y rotores de Cl(3,0)

Proyecto final de **MIA-203 Redes Neuronales y Aprendizaje Profundo** (UNI, FIIS, Unidad de Posgrado).

Julio J. Caqui Robles, Geyson A. Gonzales Cruz, Ronald E. Portocarrero Hoyos, Jhon A. Quispe Repuello y Christian Salcedo Malaver.

## Resumen

Se estudia la síntesis de nuevas vistas a partir de una sola imagen. Dada una vista fuente $I_s$ y la pose relativa $T_{rel}=T_s^{-1}T_t$, una U-Net condicionada por pose predice la vista objetivo $I_t$ de un objeto de ShapeNet-SRN Cars a 64×64. El trabajo responde dos preguntas con una ablación de igual número de parámetros (≈3,2 M):

1. ¿Mantiene el bloque de interacción geométrica de CliffordNet [1], que prescinde de la red *feed-forward* y solo se había evaluado en clasificación, su eficiencia en una tarea de predicción densa?
2. ¿Mejora la síntesis cuando la pose se codifica como un rotor $q\in\mathrm{Spin}(3)$ y se transporta el latente con el producto sándwich $z_k' = q\,z_k\,\tilde q$, en lugar de usar la matriz $[R\,|\,t]$? ¿Depende el efecto de la magnitud de la rotación?

| Modelo | Bloque | Pose |
|---|---|---|
| A (línea base) | ConvNeXt-like con FFN | matriz $[R\,|\,t]$ + FiLM |
| B | Clifford sin FFN | matriz $[R\,|\,t]$ + FiLM |
| C (propuesta) | Clifford sin FFN | rotor + FiLM + transporte sándwich |
| D (opcional) | ConvNeXt-like con FFN | rotor + FiLM + transporte sándwich |

Se reporta también la línea base trivial de copiar la vista fuente.

## Resultados

| Modelo | Parám. (M) | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Imág./s | Horas GPU |
|---|---|---|---|---|---|---|
| Copiar fuente | — | 13.64 | 0.600 | 0.249 | — | 0 |
| A Conv + matriz | 3.16 | 18.87 ± 0.00 | 0.769 ± 0.000 | 0.204 ± 0.001 | 381 | 1.37 |
| B Clifford + matriz | 3.15 | 18.54 ± 0.10 | 0.758 ± 0.005 | 0.208 ± 0.005 | 225 | 3.02 |
| C Clifford + rotor | 3.21 | 18.59 ± 0.16 | 0.758 ± 0.005 | 0.211 ± 0.006 | 225 | 3.09 |

- **Bloque (A frente a B):** el bloque de Clifford sin FFN rinde 0.33 dB menos, con ambas semillas y en el 88 % de los objetos. La brecha es nula por debajo de 30° de rotación relativa y crece hasta −0.57 dB por encima de 175°.
- **Rotor (B frente a C):** añade 0.05 dB en promedio, pero la diferencia cambia de signo entre semillas (+0.23 y −0.13 dB) y no crece con el ángulo.
- **Rotaciones cercanas a 180°:** no hay degradación pese a la discontinuidad del representante canónico del rotor.

Las métricas se calculan sobre 704 objetos de prueba con 16 vistas objetivo cada uno, y se reportan como media ± desviación estándar entre semillas. La partición de prueba solo se usa en esta evaluación final, y LPIPS se mide con VGG aunque la pérdida usa AlexNet.

## Estructura

```
clifford-nvs/
├── README.md
├── requirements.txt
├── configs/            hiperparámetros: base.yaml y un YAML por experimento
├── data/               README con origen, descarga y licencia de los datos
├── notebooks/
│   ├── 01_entrenamiento_colab.ipynb   flujo completo en Colab con guardado en Drive
│   ├── 01_entrenamiento_colab_ejecutado.ipynb   la misma ejecución con sus salidas (registro del entrenamiento reportado)
│   ├── 02_demo_inferencia.ipynb       carga de pesos e inferencia de principio a fin
│   └── 03_imagen_propia.ipynb         órbita a partir de una foto cualquiera de un auto
├── results/            tablas .tex, figuras .pdf, CSV y registros del artículo
└── src/
    ├── data.py         preprocesado, muestreo de pares en GPU, protocolo de evaluación
    ├── geometry.py     rotores de Cl(3,0), pose relativa, ángulo geodésico
    ├── models.py       bloques Conv y Clifford, codificación de pose, U-Net
    ├── metrics.py      PSNR, SSIM, LPIPS, throughput
    ├── train.py        entrenamiento con validación y reanudación
    ├── evaluate.py     evaluación en prueba, tablas y pruebas de Wilcoxon
    ├── figures.py      figuras del artículo
    ├── export.py       pesos publicables y muestra para la demostración
    ├── infer_image.py  segmentación, estimación de la vista y órbita para una imagen propia
    └── utils.py        configuración, semillas, escritura atómica
```

## Reproducción

### En Google Colab (recomendado)

Sube la carpeta del repositorio a `MyDrive/clifford-nvs` o define `REPO_URL` en la primera celda para clonarlo desde GitHub. Después abre `notebooks/01_entrenamiento_colab.ipynb`, selecciona una GPU T4 y ejecuta las celdas en orden. Los datos preprocesados, los checkpoints y los resultados se guardan en `MyDrive/clifford_nvs`. Si la sesión termina o se pierde la conexión, se vuelven a ejecutar las mismas celdas y cada run continúa desde su último checkpoint.

### En un equipo propio

```bash
git clone https://github.com/RonaldPort/clifford-nvs.git && cd clifford-nvs
pip install -r requirements.txt

# 1. Datos (ver data/README.md para obtener srn_cars.zip)
unzip srn_cars.zip -d /tmp/srn
python -m src.data --srn_root /tmp/srn --out data/processed

# 2. Entrenamiento: tres configuraciones × dos semillas
for cfg in A_conv_matrix B_clifford_matrix C_clifford_rotor; do
  for seed in 0 1; do
    python -m src.train --config configs/$cfg.yaml --seed $seed --data data/processed --work runs_root
  done
done

# 3. Evaluación: genera results/resultados.csv y las tablas .tex
CFGS="configs/A_conv_matrix.yaml configs/B_clifford_matrix.yaml configs/C_clifford_rotor.yaml"
python -m src.evaluate --configs $CFGS --seeds 0 1 --data data/processed --work runs_root --out results

# 4. Figuras
python -m src.figures --configs $CFGS --seeds 0 1 --data data/processed --work runs_root --out results \
  --focus configs/C_clifford_rotor.yaml

# 5. Análisis complementario (por semilla, por objeto, intervalos finos de ángulo, peores casos, convergencia)
python -m src.analysis --configs $CFGS --seeds 0 1 --work runs_root --out results
```

### Correspondencia entre el artículo y los comandos

| Elemento del artículo | Archivo | Comando |
|---|---|---|
| Tabla de resultados principales | `results/tabla_principal.tex` | `src.evaluate` |
| Tabla de PSNR por ángulo | `results/tabla_angulos.tex` | `src.evaluate` |
| Tabla de pruebas de Wilcoxon | `results/tabla_wilcoxon.tex` | `src.evaluate` |
| Costo computacional | `results/resultados.csv`, `results/costo.json` | `src.evaluate` |
| Curvas de entrenamiento y validación | `results/fig_curvas.pdf` | `src.figures` |
| PSNR frente al ángulo relativo | `results/fig_psnr_angulo.pdf` | `src.figures` |
| Comparación cualitativa y mapas de error | `results/fig_cualitativa.pdf`, `results/fig_error.pdf` | `src.figures` |
| Análisis de peores casos | `results/fig_peores_casos.pdf` | `src.figures` |
| Cifras por semilla, por objeto, por intervalo fino de ángulo, de peores casos y de convergencia (§IV–V) | `results/analisis.json` | `src.analysis` |

## Pesos

Los pesos de cada modelo (semilla 0) y una muestra de cinco objetos de prueba se publican en <https://drive.google.com/drive/folders/1TXojt_WquDOaJTpMon-T-5yG6f7GzgNb>. El notebook `02_demo_inferencia.ipynb` los descarga y genera una órbita completa alrededor de un objeto. La muestra deriva de SRN Cars y se comparte con fines académicos bajo los términos de uso de ShapeNet.

## Imagen propia

`notebooks/03_imagen_propia.ipynb` lleva una foto cualquiera de un auto al formato de SRN Cars y sintetiza su órbita con los tres modelos. El auto se segmenta (canal alfa, [rembg](https://github.com/danielgatis/rembg) o umbral sobre el color del borde), su punto de vista se estima por IoU entre su silueta normalizada y las de los objetos de la muestra en las 251 vistas de la espiral de prueba, con y sin reflejo, y se encuadra sobre fondo blanco a 64×64 como aparece un auto de SRN en esa vista. Las poses objetivo son las de la espiral, así que $T_{rel}$ queda dentro de la distribución de entrenamiento. La misma tarea desde la terminal:

```bash
pip install "rembg[cpu]"    # opcional
python -m src.infer_image --image foto.jpg --release release --out salida_imagen
python -m src.infer_image --image foto.jpg --release release --view 84 --obj 0 --flip 1   # vista fijada a mano
```

Genera `candidatos.png` (entrada encuadrada, silueta y vistas de referencia más parecidas), `vistas.png` (modelos × ángulo relativo), `orbita.gif` y `resumen.json`. Por la simetría del auto, una vista y su reflejo de frente a atrás pueden tener siluetas parecidas; `candidatos.png` permite confirmar la elección. Los modelos solo vieron renders sintéticos: con fotos reales la forma sigue a la pose, pero el color y los detalles se promedian.

## Reproducibilidad

- Semillas fijadas en Python, NumPy y PyTorch. El muestreo de pares usa un generador propio cuyo estado se guarda en cada checkpoint, así que un run reanudado sigue la misma secuencia que uno sin interrupciones.
- `cudnn.benchmark` está activo para acelerar el entrenamiento en la T4. Por eso pueden aparecer diferencias numéricas mínimas entre dos ejecuciones en GPU.
- `best.pt` se elige con el PSNR de 50 objetos de validación.

## Costo computacional

Todo el entrenamiento se hizo en una NVIDIA T4 de Google Colab: **7.48 horas de GPU** en total para 3 modelos × 2 semillas × 6000 pasos, unos 0.52 kWh estimados con la potencia de diseño de 70 W. No hubo pasos con pérdida no finita. Las horas de cada run están en `results/logs/*_summary.json` y el total en `results/costo.json`.

## Referencias

1. Z. Ji, “CliffordNet: All you need is geometric algebra,” arXiv:2601.06793, 2026. Código: <https://github.com/ParaMind2025/CAN>
2. H. Yang *et al.*, “Hi3D: Pursuing high-resolution image-to-3D generation with video diffusion models,” *ACM MM*, 2024.
3. V. Sitzmann, M. Zollhöfer y G. Wetzstein, “Scene representation networks,” *NeurIPS*, 2019.
4. A. Yu, V. Ye, M. Tancik y A. Kanazawa, “pixelNeRF: Neural radiance fields from one or few images,” *CVPR*, 2021.
5. O. Ronneberger, P. Fischer y T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” *MICCAI*, 2015.
6. R. Zhang *et al.*, “The unreasonable effectiveness of deep features as a perceptual metric,” *CVPR*, 2018.

## Licencia

El código se publica bajo licencia MIT (ver `LICENSE`). Los datos están sujetos a los términos de ShapeNet (ver `data/README.md`). La implementación del bloque de interacción se basa en el repositorio oficial de CliffordNet, también bajo licencia MIT.
