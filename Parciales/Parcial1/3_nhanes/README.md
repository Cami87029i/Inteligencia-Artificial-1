DESCRIPCIÓN DEL DATASET:

NHANES es una encuesta nacional representativa de salud y nutrición de EE.UU. administrada por el CDC. El archivo de **demografía** (`DEMO_L`) contiene información sociodemográfica de todos los participantes de la muestra del ciclo 2021-2023.
**Valores codificados:** Las variables categóricas usan códigos numéricos (e.g., 1=Masculino, 2=Femenino). Los valores `7`, `9`, `77`, `99`, `777`, `999` generalmente indican **'Refused'** o **'Don't Know'** y deben tratarse como `NaN`. En bases de datos gigantescas como NHANES, no se guarda texto (como "Masculino" o "Casado") por varias razones:
 - Eficiencia de espacio: Guardar un 1 ocupa muchísimo menos espacio en el servidor que guardar la palabra "Masculino". Multiplicado por miles de personas y variables, la diferencia es enorme.
 - Estandarización: Evita errores de dedo. Si permitieran texto, alguien podría escribir "Masculino", otro "masculino", otro "Masc", y la IA pensaría que son cosas distintas. Al usar el código 1, no hay pérdida de interpretación.
 - Compatibilidad Matemática:Los algoritmos de IA (como las redes neuronales) solo entienden números. No se pueden multiplicar cadenas por pesos, pero sí números por pesos.
### Variables codificadas como missing en NHANES
Los siguientes valores deben convertirse a `NaN` antes de cualquier análisis:
- `7`, `9` → 'Refused' / 'Don't know' (variables de 1 dígito)
- `77`, `99` → ídem (2 dígitos)
- `777`, `999` → ídem (3 dígitos)
Se toman los siguientes números por lo siguiente:
- El código 7 (Refused): La persona no quiso contestar. Es un dato sensible (ej. ingresos o drogas).
- El código 9 (Don't Know): La persona no sabe la respuesta (ej. no sabe si su abuelo tuvo diabetes).
  La razón por las que se los convierte a NAN es por que, por ejemplo, si se dejara un 99 en la columna de "años viviendo en EEUU", el modelo que estamos entrenando pensará que la persona realmente lleva viviendo 99 años viviendo ahi, lo cuál puede provocar que se "ensucie" el promedio y el modelo se confunde, al pasarlos por NAN, se le dice a la IA que "ignore esa celda" porque la información no es confiable.
- ### ¿Cuál es el target (y)?
El archivo de demografía **no tiene un target predefinido** — es un dataset de features que se **combina con otros módulos** de NHANES (laboratorio, examinación, cuestionarios). Los targets típicos en NHANES son:
- `BMXBMI` (IMC) — del módulo de examinación `BMX_L`
- `LBXGH` (HbA1c, diabetes) — del módulo de laboratorio
- `BPQ020` (hipertensión diagnosticada) — del cuestionario
> Para este cuadernillo usaremos **INDFMPIR** (ratio pobreza) como target de ejemplo para demostrar la preparación. Lo usamos porque está normalizado y es un gran predictor debido a que si se logra predecir el ratio de pobreza se puede entender mejor el riesgo de desnutrición o falta de acceso a medicina.
