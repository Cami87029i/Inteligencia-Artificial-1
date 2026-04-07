**Variables Clave (Predictores):**

**RIAGENDR (Género):** Sexo biológico del participante (1 para Masculino, 2 para Femenino). Es una variable categórica fundamental.

**RIDAGEYR (Edad):** Edad en años al momento de la entrevista. Es una variable continua (0 a 80). Nota: Los mayores de 80 aparecen todos como "80" para proteger su privacidad.

**RIDRETH3 (Raza/Etnicidad):** Clasificación étnica detallada (incluyendo población asiática). Es mejor usar esta que la versión antigua (RIDRETH1).

**DMDEDUC2 (Educación Adultos):** Nivel educativo para personas de 20 años o más. Es una variable ordinal (del 1 al 5).

**DMDMARTZ (Estado Civil):** Indica si la persona está casada, viuda, divorciada, separada o soltera.

**INDFMPIR (Ratio de Pobreza):** Es un valor continuo que divide los ingresos del hogar entre la línea de pobreza. Es mejor que el ingreso bruto para modelos de IA porque está normalizado.

**DMDBORN4 (País de Nacimiento):** Indica si nació en EE.UU. o en otro país.

**Variables Técnicas (Se eliminan):**

Estas columnas no aportan conocimiento al aprendizaje del modelo y suelen causar ruido.

**SEQN (ID del participante):**Es solo un número de secuencia. No tiene relación con la salud o el comportamiento, por lo que se elimina.

**SDDSRVYR (Ciclo):** Como todos tus datos son del mismo periodo (2021-2023), esta columna siempre vale "12". Al ser una constante, no sirve para la IA.

**SDDSRVYR (Ciclo):** Como todos tus datos son del mismo periodo (2021-2023), esta columna siempre vale "12". Al ser una constante, no sirve para la IA.



