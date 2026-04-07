DESCRIPCIÓN DEL DATASET:
- 17,389 registros, 17 columnas, cero nulos 
- Se eliminan: instant (índice), dteday (ya está en yr/mnth/weekday), casual y registered (data leakage —suman exactamente cnt)
- Target: cnt — total de alquileres por hora
-Columna más importante: hr (hora del día), con picos clarísimos a las 8am y 17-18h
