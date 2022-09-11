# Talent Squad - Data Science II - Nuwe

## Sobre el desafio

Reto NUWE clasificación de imágenes que en clasificar a partir de una imagen a qué tipo de deporte corresponde, teniendo los siguientes: baseball, cricket y fútbol.
[Talent Squad - Data Science II](https://nuwe.io/dev/challenges/talent-squad-data-science-ii)
La solución se evaluará con los criterios:
Objetivos
1. Aumento de muestras en las imágenes
2. Calcular el F1-score macro.

## Solución
Se ha utilizado [fastai](https://www.fast.ai/) para crear el clasificador dado su facilidad de uso y su buen rendimiento con sets de imágenes reducidos.

Todo el entrenamiento del modelo está en el notebook [`train.ipynb`](train.ipynb). Para el modelo se ha utilizado la arquitectura resnet34. Se han probado a hacer data augmetation que ofrece fastai:
```Python
get_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=360,
    max_warp=0,
    max_zoom=1.1,
    max_lighting=0.1,
    p_lighting=0.5
)
```
pero no se ha logrado una f1-score mejor que sin realizar data augmentation. Por eso se ha descartado.
El F1-score macro es de 0.93. (ver notebook [`train.ipynb`](train.ipynb))

Para ejecutar los notebooks se puede preparar el entorno con el comando siguiente teniendo el conda instalado:
```Bash
conda env create -f env.yaml
```

## Resultados
Los resultados se pueden ver en los notebooks [`train.ipynb`](train.ipynb) y [`inference.ipynb`](inference.ipynb). El primero tiene el código para entrenar el modelo y el segundo utiliza el modelo entrenado para ver resultados sobre el test set y generar el fichero [`predictions.csv`](predictions.csv) con las predicciones para el test set.

