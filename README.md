# IAS_Autonomous_Driving_Car
IAS - ENIB : School Proyect to program an autonomus driving car

# Instalar dependencias desde el requirements
pip install -r requirements.txt


# Cosas a hacer:

Pretratamiento CSV:
1. Poner en 0 las velocidades en las entradas donde los GPIO estén todos en 0
2. Generar data artificial (usando interpolaciones) para poder asociarle una entrada del .csv a cada imagen
3. Asociar entradas del .csv con su imágen correspondiente


# Propuesta:
Creamos un script de python que genere un nuevo CSV global que reuna la informacion de las carpetas de la base de datos, entre los csv y las imagenes. El formato sera igual que el de los CSV con una columna con las direcciones asociadas a cada imagen.

Si alguna fila dice que TODOS sus GPIO estan en 0, y la velocidad esta en un valor distinto de 0 ya sea para motor A o motor B, es un error del codigo y debera ser corregido cambiando la velocidad de motor A y motor B a 0.
Esto es asi ya que el dirver internamente FRENA el motor si todos los GPIO de sentido estan en 0.

 Arrancamos asociando la imagen anterior de un comando mas cercana al momento del comando. Dado que existen muchas mas imagenes que comandos CSV van a quedar imagenes no asociadas a ningun comando. Para estas imagenes no asociadas vamos a interpolar todos los campos del comando (slavo el tiempo) entre comando real y comando real, dependiendo de cuantas imagenes sin asociar existen entre estos 2 comandos. Utilizaremos el tiempo de la imagen como parametro de entrada para interpolar los campos. Es decir tengo una imagen llamada 1 (tomada en el ms 1) y tenemos un comando hecho en ms 2, el proximo comando esta hecho en ms 5. Existen imagen 1, imagen 3 imagen 4. El comando hecho en ms 2 es asociado a la imagen anterior (es decir 1), imagen 4 es asociada al comando de ms 5. imagen 3 no tiene comando asociado asique interpolamos todos los parametros del comando utilizando el comando en ms 2 y en ms 5 como parametros de borde para la interpolacion. 



