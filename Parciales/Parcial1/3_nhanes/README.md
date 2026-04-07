DESCRIPCIÓN DEL DATASET:
 Encuesta nacional de salud y nutrición de EE.UU. Datos demográficos, laboratorio, exámenes físicos y cuestionarios. Disponible por ciclos de 2 años. Formato XPT (SAS Transport).
-Tiene ~46 columnas pero muchas son redundantes o de diseño muestral (pesos, estratos) — se seleccionan 10 relevantes
-Cuidado especial: los valores 7, 9, 77, 99, 777, 999 significan "Refused/Don't Know" y deben reemplazarse por NaN antes de todo
-No tiene un target propio — es un archivo de features. Se usa INDFMPIR (ratio ingreso/pobreza) como target de ejemplo
-El split aquí sí es aleatorio porque no es serie de tiempo
