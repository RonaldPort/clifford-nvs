# Datos: ShapeNet-SRN Cars

## Origen

El conjunto proviene de *Scene Representation Networks* (Sitzmann et al., NeurIPS 2019). Los autores renderizaron la categoría *cars* de ShapeNet Core v2 (Chang et al., 2015) a 128×128 píxeles. Cada imagen va acompañada de la pose de su cámara. Es el estándar de síntesis de nuevas vistas a partir de una sola imagen que usan pixelNeRF, Zero-1-to-3 y trabajos posteriores.

| Partición | Objetos | Vistas por objeto | Distribución de cámaras |
|---|---|---|---|
| `cars_train` | 2458 | 50 | aleatoria sobre la esfera |
| `cars_val` | 352 | 251 | espiral arquimediana |
| `cars_test` | 704 | 251 | espiral arquimediana |

Los objetos de las tres particiones son disjuntos, de modo que no hay fuga entre entrenamiento, validación y prueba. `python -m src.data` imprime los conteos reales al preprocesar.

## Descarga

pixelNeRF distribuye el archivo `srn_cars.zip` en una carpeta pública de Google Drive:

<https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR>

Lo más fiable en Colab es abrir el enlace y agregar `srn_cars.zip` a *Mi unidad* con «Agregar acceso directo». El notebook `notebooks/01_entrenamiento_colab.ipynb` lo busca ahí. Si no lo encuentra, lo descarga con `gdown`, aunque esa descarga puede fallar cuando el archivo supera su cuota diaria.

## Estructura original

```
cars_<train|val|test>/<id_shapenet>/
    rgb/000000.png ...        imagen 128×128
    pose/000000.txt ...       matriz 4×4 cámara→mundo (16 valores)
    intrinsics.txt            no se usa
```

## Preprocesado

```bash
python -m src.data --srn_root /ruta/descompresion --out data/processed --res 64 --val_objects 100
```

Cada imagen se compone sobre fondo blanco y se reduce a 64×64 con filtro Lanczos. Cada partición se guarda como un `.npz` con las imágenes en `uint8`, las poses y los índices de cada objeto. La validación conserva 100 objetos, de los cuales el entrenamiento usa 50 para elegir `best.pt`. La prueba se conserva completa.

| Archivo | Tamaño aproximado |
|---|---|
| `cars_train_64.npz` | 1,5 GB |
| `cars_val_64.npz` | 0,3 GB |
| `cars_test_64.npz` | 2,2 GB |

Los `.npz` no se suben al repositorio (ver `.gitignore`).

## Licencia

ShapeNet se distribuye bajo sus [términos de uso](https://shapenet.org/terms), que permiten el uso no comercial con fines de investigación y educación. Los renders de SRN heredan esas condiciones. Por eso este repositorio no redistribuye los datos y solo enlaza su fuente original.
