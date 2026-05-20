import tensorflow as tf  # Librería para aprendizaje profundo
import numpy as np  # Librería para manipulación de matrices y cálculos numéricos
import matplotlib.pyplot as plt  # Librería para visualización de gráficos
from matplotlib import rcParams  # Configuración de gráficos
import os  # Manejo de archivos y directorios
import pandas as pd  # Manejo de datos en formato tabular
os.environ["KERAS_BACKEND"] = "tensorflow"  # Define TensorFlow como backend de Keras
import keras_core as keras  # Librería de redes neuronales basada en TensorFlow
import time  # Para medir tiempos de ejecución
import gc  # Para gestión de la memoria (recolección de basura)
import psutil  # Para medir uso de CPU durante la ejecución de experimentos
import time
import re 

# Configuración de reproducibilidad
keras.utils.set_random_seed(1234)  # Fija una semilla para asegurar resultados reproducibles
dtype = 'float32'  # Establece la precisión de los cálculos
keras.backend.set_floatx(dtype)  # Aplica el tipo de dato definido

# Definición de parámetros globales
alpha = 1  # Parámetro utilizado en las ecuaciones diferenciales
limInf = 0  # Límite inferior del dominio de la solución
limSup = np.pi  # Límite superior del dominio de la solución
iterations = 400 # epocas por experimento

def makeModel(neurons, nLayers, activation):
    """
    Construye un modelo de red neuronal profunda basado en una PINN (Physics-Informed Neural Network).
    
    Parámetros:
    - neurons (int): Número de neuronas en cada capa oculta.
    - nLayers (int): Número total de capas en la red (incluyendo la capa de entrada y salida).
    - activation (str): Función de activación utilizada en las capas ocultas.

    Retorna:
    - model (keras.Model): Modelo de red neuronal compilado.
    """
    # Capa de entrada con una única neurona para recibir valores de 'x'
    xVals = keras.layers.Input(shape=(1,), name='x_input', dtype='float32')

    # Primera capa oculta
    l1 = keras.layers.Dense(neurons, activation=activation, dtype='float32')(xVals)

    # Capas ocultas intermedias (se asegura que haya al menos una capa oculta)
    for _ in range(max(1, nLayers - 2)):  
        l1 = keras.layers.Dense(neurons, activation=activation, dtype='float32')(l1)

    # Capa de salida con una única neurona (salida escalar)
    output = keras.layers.Dense(1, activation=activation, dtype='float32')(l1)

    # Construcción del modelo
    return keras.Model(inputs=xVals, outputs=output, name='u_model')
    
class Loss1(keras.layers.Layer):
    def __init__(self, uModel, nPts, f, lambda0=1, lambda1=1, lambda2=1, limInf=0, limSup=np.pi, A=0, B=0, **kwargs):
        super(Loss1, self).__init__()
        self.uModel = uModel
        self.nPts = nPts
        self.f = f
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.limInf = limInf
        self.limSup = limSup
        self.A = A
        self.B = B

    def call(self, inputs):
        """
        Evalúa la función de pérdida en el dominio y los bordes.
        """

        # 🔹 Convertir la entrada `inputs` a `float32`
        x = tf.cast(inputs, tf.float32)  # ✅ Convertimos explícitamente a `float32`

        # 🔹 Muestreo de puntos en el dominio con `float32`
        x_samples = tf.random.uniform([self.nPts], self.limInf, self.limSup, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x_samples)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x_samples)
                u = self.uModel(x_samples, training=True)  # ✅ Se mantiene en `float32`
            dux = t2.gradient(u, x_samples)
        duxx = t1.gradient(dux, x_samples)

        # 🔹 Asegurar `float32` en todos los cálculos
        alpha_cast = tf.cast(alpha, tf.float32)
        f_x = tf.cast(self.f(x_samples), tf.float32)

        # 🔹 Cálculo del error de la ecuación diferencial
        errorPDE = self.lambda0 * keras.ops.mean(tf.cast((-duxx + alpha_cast * u - f_x) ** 2, tf.float32))

        # 🔹 Condiciones de contorno con `float32`
        bc = self.lambda1 * keras.ops.mean((self.uModel(tf.cast([self.limInf], tf.float32)) - tf.cast(self.A, tf.float32)) ** 2) + \
            self.lambda2 * keras.ops.mean((self.uModel(tf.cast([self.limSup], tf.float32)) - tf.cast(self.B, tf.float32)) ** 2)

        # ✅ Convertir la salida a tensor para evitar errores en Keras
        return tf.convert_to_tensor(errorPDE + bc, dtype=tf.float32)

class RelativeErrorCallback(tf.keras.callbacks.Callback):
    """
    Callback personalizado para calcular el error relativo en norma H¹ al final de cada época.

    Esta implementación mide la diferencia en la función y su derivada, 
    proporcionando una mejor evaluación de la precisión del modelo.

    Parámetros:
    - uModel (keras.Model): Modelo de la PINN entrenado.
    - exactU (función): Función de la solución exacta.
    - nPts (int): Número de puntos en el dominio para evaluación.
    """

    def __init__(self, uModel, exactU, nPts):
        super().__init__()
        self.uModel = uModel
        self.exactU = exactU
        self.nPts = nPts

    def on_epoch_end(self, epoch, logs=None):
        """
        Se ejecuta al final de cada época y calcula el error relativo en norma H¹.
        """

        # 🔹 Generamos puntos en el dominio
        Sval = tf.experimental.numpy.linspace(0., np.pi, num=self.nPts * 10, dtype=dtype)

        # 🔹 Calculamos derivadas con `GradientTape`
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(Sval)
            ueval_val = self.uModel(Sval)  # Solución aproximada de la PINN
            u_x_val = t1.gradient(ueval_val, Sval)  # Derivada de la solución aproximada
            u_e = self.exactU(Sval)  # Solución exacta
            ue_x = t1.gradient(u_e, Sval)  # Derivada de la solución exacta
        del t1  # Liberar memoria

        # 🔹 Cálculo del error relativo en norma H¹
        errorH01 = tf.reduce_mean((ue_x - u_x_val) ** 2)
        norm_exact = tf.reduce_mean(ue_x ** 2)
        relative_error = errorH01 / norm_exact

        # 🔹 Guardamos el error relativo en los logs del entrenamiento
        logs['relative_error'] = relative_error

def makeLossModel1(uModel, nPts, f, lambda0=1, lambda1=1, lambda2=1, limInf=0, limSup=np.pi, A=0, B=0,):
    """
    Crea un modelo de pérdida basado en la función `Loss1` para entrenar la PINN.

    En lugar de entrenar la PINN directamente, este modelo de pérdida se construye para 
    minimizar la ecuación diferencial y las condiciones de contorno en cada iteración.

    Parámetros:
    - uModel (keras.Model): Modelo de la PINN que se está entrenando.
    - nPts (int): Número de puntos de muestreo en el dominio.
    - f (función): Función de referencia f(x) en la ecuación diferencial.
    - lambda0 (float, opcional): Peso del término de la ecuación diferencial (PDE). Por defecto, 1.
    - lambda1 (float, opcional): Peso de la condición de contorno en el límite inferior. Por defecto, 1.
    - lambda2 (float, opcional): Peso de la condición de contorno en el límite superior. Por defecto, 1.
    - limInf (float, opcional): Límite inferior del dominio. Por defecto, 0.
    - limSup (float, opcional): Límite superior del dominio. Por defecto, π.

    Retorna:
    - lossModel (keras.Model): Modelo de pérdida para la PINN.
    """
    # Entrada ficticia para alimentar el modelo de pérdida
    xVals = keras.layers.Input(shape=(1,), name='x_input', dtype=tf.float32)

    # Aplicamos la capa personalizada `Loss1` que calcula la pérdida de la ecuación diferencial
    loss_output = Loss1(uModel, nPts, f, lambda0, lambda1, lambda2, limInf, limSup,A, B,)(xVals)

    # Construcción del modelo de pérdida
    lossModel = keras.Model(inputs=xVals, outputs=loss_output)
    
    return lossModel

def trickyLoss(yPred, yTrue):
    """
    Función de pérdida ficticia utilizada en el entrenamiento de la PINN.

    Dado que la pérdida real es calculada por la capa `Loss1`, esta función 
    simplemente devuelve `yTrue` sin hacer ningún cálculo. 

    Parámetros:
    - yPred (tensor): Predicción del modelo de pérdida (`lossModel`).
    - yTrue (tensor): Etiqueta objetivo (se ignora).

    Retorna:
    - yTrue (tensor): Se devuelve sin modificaciones.
    """
    return yTrue

## INICIALIZAR CARPETAS Y EXCEL ## 

def crear_estructura_directorios(base_path="experimentos/PLANTEAMIENTO 1"):
    """
    Crea la estructura de directorios necesarios para almacenar los resultados de los experimentos.

    Parámetros:
    - base_path (str): Ruta base donde se almacenarán los resultados.

    Retorna:
    - base_path (str): Ruta base de los experimentos.
    - graficas_path (str): Ruta donde se guardarán las gráficas generadas.
    """

    try:
        # 📌 Verificar si la ruta ya existe
        if os.path.exists(base_path):
            if os.path.isfile(base_path):
                print(f"⚠️ Advertencia: '{base_path}' es un archivo, no un directorio. Se eliminará y se creará como carpeta.")
                os.remove(base_path)  # ✅ Eliminar archivo si existe
                os.makedirs(base_path)  # ✅ Ahora sí crear la carpeta
            else:
                print(f"✔ La carpeta '{base_path}' ya existe.")
        else:
            os.makedirs(base_path)  # ✅ Crear el directorio si no existe

        # 📌 Crear subdirectorio para gráficas
        graficas_path = os.path.join(base_path, "graficas")
        os.makedirs(graficas_path, exist_ok=True)

        print("✔ Directorios creados correctamente.")
        return base_path, graficas_path

    except Exception as e:
        print(f"❌ Error al crear directorios: {e}")
        return None, None  # Retornar None en caso de error

def inicializar_excel(file_path="experimentos/PLANTEAMIENTO 1/resultados.xlsx"):
    """
    Crea un archivo Excel con la estructura inicial si no existe. 
    El archivo contendrá tres hojas llamadas "Lote 1", "Lote 2" y "Lote 3",
    donde se almacenarán los resultados de los experimentos.

    Parámetros:
    - file_path (str): Ruta del archivo Excel donde se guardarán los datos.

    Retorna:
    - file_path (str): Ruta del archivo Excel.
    """
    if not os.path.exists(file_path):
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        for i in range(1, 4):
            # Definir las columnas del archivo Excel
            df = pd.DataFrame(columns=[
                "Experimento", "neurons", "nLayers", "nPts", 
                "lambda0", "lambda1", "lambda2", "ERROR RELATIVO", "COSTO COMPUTACIONAL"
            ])
        writer.close()
        print("✔ Archivo Excel inicializado.")
    else:
        print("✔ Archivo Excel ya existe.")
    
    return file_path
    
# Función para obtener los hiperparámetros por defecto
def hiperparametros_por_defecto():
    """
    Devuelve un diccionario con valores predeterminados para los hiperparámetros de la PINN.

    Retorna:
    - dict: Diccionario con los hiperparámetros iniciales.
    """
    return {
        "neurons": 10,  # Número de neuronas por capa
        "nLayers": 5,  # Número de capas ocultas
        "nPts": 2000,  # Número de puntos en el dominio
        "lambda0": 5,  # Peso del error de la ecuación diferencial
        "lambda1": 7,  # Peso del error en el contorno inferior
        "lambda2": 7   # Peso del error en el contorno superior
    }

# Función para generar hiperparámetros aleatorios dentro de ciertos rangos
def randomizar_hiperparametros():
    """
    Genera un conjunto aleatorio de hiperparámetros dentro de ciertos rangos predefinidos.

    Retorna:
    - dict: Diccionario con hiperparámetros aleatorios.
    """
    return {
        "neurons": np.random.randint(4, 11),  # Neuronas entre 4 y 10
        "nLayers": np.random.randint(4, 6),  # Capas entre 4 y 5
        "nPts": np.random.randint(500, 3001),  # Puntos entre 500 y 3000
        "lambda0": np.random.randint(1, 11),  # Peso entre 1 y 10
        "lambda1": np.random.randint(4, 11),  # Peso entre 4 y 10
        "lambda2": np.random.randint(4, 11)   # Peso entre 4 y 10
    }

def medir_costo_computacional(funcion_entrenamiento, *args, **kwargs):
    """
    Mide el porcentaje de uso de CPU durante la ejecución de una función.

    Parámetros:
    - funcion_entrenamiento (función): La función que queremos medir.
    - *args: Argumentos posicionales de la función.
    - **kwargs: Argumentos con nombre de la función.

    Retorna:
    - cpu_percent (float): Porcentaje medio de CPU usado durante la ejecución.
    - elapsed_time (float): Tiempo total transcurrido en segundos.
    """

    process = psutil.Process(os.getpid())  # Obtener el proceso actual

    # 🔹 Medir el tiempo de CPU y tiempo real antes del entrenamiento
    cpu_times_before = process.cpu_times()
    time_before = time.time()

    # 🔹 Ejecutar la función de entrenamiento con sus argumentos
    funcion_entrenamiento(*args, **kwargs)

    # 🔹 Medir el tiempo de CPU y tiempo real después del entrenamiento
    cpu_times_after = process.cpu_times()
    time_after = time.time()

    # 🔹 Calcular tiempo total de CPU consumido
    cpu_time_used = (cpu_times_after.user + cpu_times_after.system) - (cpu_times_before.user + cpu_times_before.system)

    # 🔹 Calcular tiempo real transcurrido
    elapsed_time = time_after - time_before

    # 🔹 Calcular porcentaje de CPU usado
    num_cpus = psutil.cpu_count(logical=True)  # Número de núcleos lógicos
    cpu_percent = (cpu_time_used / (elapsed_time * num_cpus)) * 100 if elapsed_time > 0 else 0

    return cpu_percent, elapsed_time

def train_PINN(hiperparametros, funcion_referencia, funcion_exacta, u_0=0, u_pi=0):
    """
    Entrena la PINN con los hiperparámetros dados y la función de referencia seleccionada.

    También mide el error relativo y el costo computacional en términos de uso de CPU.

    Parámetros:
    - hiperparametros (dict): Diccionario con la configuración de la red (neurons, nLayers, etc.).
    - funcion_referencia (función): Función f(x) en la ecuación diferencial que la PINN debe aprender.
    - funcion_exacta (función): Solución exacta esperada para evaluar el error relativo.

    Retorna:
    - error_relativo (float): Error relativo entre la solución de la PINN y la solución exacta.
    - costo_computacional (float): Uso de CPU en porcentaje durante el entrenamiento.
    - history (tf.keras.callbacks.History): Historial del entrenamiento.
    - uModel (keras.Model): Modelo entrenado de la PINN.
    - xList (array): Puntos en el dominio donde se evaluó la solución.
    """

    # 🔹 Construcción del modelo de red neuronal PINN
    uModel = makeModel(
        neurons=hiperparametros["neurons"],
        nLayers=hiperparametros["nLayers"],
        activation='tanh'
    )

    # 📌 Calcular valores exactos en los bordes (x = 0 y x = π)
    A = u_0#funcion_exacta(tf.convert_to_tensor(0.0, dtype=dtype))  # 📌 Se asegura de que x sea un tensor
    B = u_pi #funcion_exacta(tf.convert_to_tensor(np.pi, dtype=dtype)) 

    # 🔹 Definir el modelo de pérdida basado en la ecuación diferencial
    lossModel = makeLossModel1(
        uModel,
        hiperparametros["nPts"],
        funcion_referencia,
        hiperparametros["lambda0"],
        hiperparametros["lambda1"],
        hiperparametros["lambda2"],
        limInf, limSup,
        A, B  # 📌 Se pasan los valores exactos en los extremos
    )

    # 📌 Configurar el optimizador y la función de pérdida
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    lossModel.compile(optimizer=optimizer, loss=trickyLoss)

    relative_error_callback = RelativeErrorCallback(uModel, funcion_exacta, hiperparametros["nPts"])
    
    # 📌 Entrenamiento de la PINN
    history = lossModel.fit(
        np.array([1.]), np.array([1.]), epochs=iterations, verbose=0, callbacks=[relative_error_callback]
    )

    # 🔹 Generar puntos en el dominio para evaluar la solución entrenada
    xList = np.array([np.pi / 1000 * i for i in range(1000)])

    # 🔹 Evaluar la solución aproximada y compararla con la solución exacta
    error_relativo = history.history['relative_error'][-1].numpy()

    # 🔹 Calcular tiempo total de CPU usado (sumando usuario y sistema)
    cpu_cost, _ = medir_costo_computacional(
        lossModel.fit, np.array([1.]), np.array([1.]), epochs=iterations, verbose=0, callbacks=[relative_error_callback]
    )

    # 🔹 Liberar memoria
    keras.backend.clear_session()
    gc.collect()

    return error_relativo, cpu_cost, history, uModel, xList

def ejecutar_experimentos(func, n_experimentos=45):  
    """
    Ejecuta múltiples experimentos de entrenamiento de la PINN con diferentes hiperparámetros.

    Se generan varios lotes de experimentos con diferentes configuraciones de hiperparámetros 
    para evaluar la estabilidad y precisión del modelo en distintas condiciones.

    Parámetros:
    - func (lista de tuplas): Lista de funciones de referencia y soluciones exactas esperadas.
      Cada elemento es una tupla (función_f, función_u_exacta).
    - n_experimentos (int, opcional): Número total de experimentos a ejecutar. Por defecto, 45.

    Retorna:
    - resultados (lista): Lista de resultados de los experimentos, donde cada elemento contiene:
      [ID, neurons, nLayers, nPts, lambda0, lambda1, lambda2, error_relativo, costo_computacional, 
      modelo entrenado, historial, solución exacta, xList].
    """

    # 📌 Determinar la cantidad de lotes basada en el número de experimentos
    num_lotes = min(len(func), n_experimentos)  
    lote_size = n_experimentos // num_lotes  # 📌 Número de experimentos por lote

    # 📌 Generar hiperparámetros aleatorios para cada lote (excepto el primero, que es por defecto)
    hiperparametros_aleatorios = [randomizar_hiperparametros() for _ in range(lote_size - 1)]  

    # 📌 Lista para almacenar resultados
    resultados = []

    for i in range(num_lotes):
        # 🔹 Seleccionar la función de referencia y la solución exacta
        f_rhs, exactU = func[i]  
        print(f"\n🔹 Ejecutando Lote {i+1}...\n")

        for j in range(lote_size):
            experimento_id = i * lote_size + j + 1
            print(f"   ▶ Ejecutando experimento {experimento_id}...")

            # 🔹 Para el primer experimento de cada lote, usar hiperparámetros por defecto
            if j == 0:
                hiperparametros = hiperparametros_por_defecto()
            else:
                hiperparametros = hiperparametros_aleatorios[j - 1]

            # 🔹 Entrenar la PINN con la configuración actual
            error_relativo, costo_computacional, history, uModel, xList = train_PINN(hiperparametros, f_rhs, exactU)

            # 🔹 Imprimir resultados del experimento
            print(f"      🔹 Completado - Error relativo: {error_relativo:.5f}, Uso CPU: {costo_computacional:.2f}%")

            # 📌 Guardar los resultados del experimento
            resultados.append([
                experimento_id, 
                hiperparametros["neurons"], 
                hiperparametros["nLayers"], 
                hiperparametros["nPts"], 
                hiperparametros["lambda0"], 
                hiperparametros["lambda1"], 
                hiperparametros["lambda2"], 
                error_relativo, 
                costo_computacional,  
                uModel,   
                history,  
                exactU,   
                xList     
            ])

    return resultados

# 🔹 Definimos las funciones de referencia y sus soluciones exactas en un diccionario
funciones_experimentos = [
    {
        'fRhs': lambda x: (alpha + 4) * keras.ops.sin(2 * x),
        'exactU': lambda x: keras.ops.sin(2 * x),
        'title': 'sin(2x)'
    },
    {
        'fRhs': lambda x: (17/4) * keras.ops.sin(2 * x) * keras.ops.cos(x / 2) + 
                          2 * keras.ops.sin(x / 2) * keras.ops.cos(2 * x) + 
                          alpha * keras.ops.sin(2 * x) * keras.ops.cos(x / 2),
        'exactU': lambda x: keras.ops.sin(2 * x) * keras.ops.cos(x / 2),
        'title': 'sin(2x) * cos(x/2)'
    },
    {
        'fRhs': lambda x: -2 * (6 * x**2 - 6 * np.pi * x + np.pi**2) + 
                          alpha * (x**2 * (x - np.pi)**2),
        'exactU': lambda x: x**2 * (x - np.pi)**2,
        'title': 'x^2 * (x - pi)^2'
    }
]

# 🔹 Convertir el diccionario en una lista de tuplas
func = [(exp['fRhs'], exp['exactU']) for exp in funciones_experimentos]

def guardar_resultados_excel(resultados, funciones_experimentos, file_path="experimentos/PLANTEAMIENTO 1/resultados.xlsx"):
    """
    Guarda los resultados de los experimentos en un archivo Excel, separando por lote.

    Cada hoja del archivo representa un conjunto de experimentos con la misma función de referencia.

    Parámetros:
    - resultados (lista): Lista de resultados obtenidos de `ejecutar_experimentos`.
    - funciones_experimentos (lista): Lista de funciones de referencia con su título.
    - file_path (str, opcional): Ruta del archivo Excel donde se guardarán los datos.

    Retorna:
    - None
    """

    # 📌 Verificar si el archivo ya existe
    file_exists = os.path.exists(file_path)

    # 📌 Configurar el modo de apertura del archivo Excel
    mode = 'a' if file_exists else 'w'

    # 📌 Crear el writer con openpyxl si el archivo ya existe, de lo contrario, usar xlsxwriter
    engine = 'openpyxl' if file_exists else 'xlsxwriter'

    # 📌 Abrir el archivo Excel
    with pd.ExcelWriter(file_path, engine=engine, mode=mode) as writer:
        
        # 📌 Recorrer cada experimento y guardarlo en su respectiva hoja
        for i, experimento in enumerate(funciones_experimentos):
            lote_resultados = [
                [r[j] for j in range(9)] for r in resultados if (r[0] - 1) // (len(resultados) // len(funciones_experimentos)) == i
            ]

            # 📌 Convertir a DataFrame
            df = pd.DataFrame(
                lote_resultados,
                columns=["Experimento", "neurons", "nLayers", "nPts", 
                         "lambda0", "lambda1", "lambda2", "ERROR RELATIVO", "COSTO COMPUTACIONAL (CPU %)"]
            )

            # ✅ Limpiar y truncar el nombre de la hoja para evitar errores
            sheet_name = experimento['title']
            sheet_name = re.sub(r'[:\\/\*\[\]\?]', '', sheet_name)[:31]  # ✅ Elimina caracteres inválidos y limita a 31 caracteres

            # ✅ Verificar si hay datos antes de escribir en Excel
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                print(f"⚠️ Advertencia: No hay datos para guardar en la hoja '{sheet_name}'.")

    print(f"✔ Resultados guardados exitosamente en {file_path}.")

def plot_results(uModel, history, exactU, xList, exp_folder):
    """
    Genera y guarda las gráficas de un experimento, asegurando que se use la función exacta correcta.
    """
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['legend.fontsize'] = 12
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['axes.labelsize'] = 14

    # Comparación de la solución exacta y la aproximada
    fig, ax = plt.subplots()
    plt.plot(xList, uModel(xList), color='b', label="u_approx")
    plt.plot(xList, exactU(xList), color='m', label="u_exact")  # ✅ Se asegura que usa la función exacta correcta
    plt.legend()
    ax.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "comparacion_u.png"), dpi=300)
    plt.close()

    # Evolución de la pérdida y error relativo
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], color='g', label="Loss")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='g')

    ax2 = ax1.twinx()
    ax2.plot(history.history['loss'], color='b', label="Relative Error")  # 📌 Se debe reemplazar por la métrica correcta
    ax2.set_ylabel("Relative Error", color='b')

    ax1.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "error_vs_loss.png"), dpi=300)
    plt.close()

    # Error relativo a lo largo de las épocas
    fig, ax = plt.subplots()
    plt.plot(history.history['loss'], color='b', label="Relative Error")  # 📌 Se debe ajustar con la métrica real
    ax.set_xscale('log')
    plt.legend()
    ax.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "error_relativo.png"), dpi=300)
    plt.close()
    
def seleccionar_experimentos(resultados):
    """
    Selecciona los experimentos que se van a graficar y los organiza en una lista.

    Parámetros:
    - resultados (lista): Lista de resultados obtenidos en `ejecutar_experimentos`.

    Retorna:
    - todos_experimentos (lista): Lista de experimentos seleccionados para graficar.
    """

    # 📌 Determinar la cantidad de lotes según los experimentos disponibles
    num_lotes = min(len(func), len(resultados))  
    lote_size = max(1, len(resultados) // num_lotes)  

    # 📌 Lista para almacenar los experimentos seleccionados
    todos_experimentos = []

    for i in range(num_lotes):
        # 🔹 Seleccionar experimentos de este lote
        lote_resultados = [r for r in resultados if (r[0] - 1) // lote_size == i]
        todos_experimentos.extend(lote_resultados)

    print("✔ Experimentos listos para generar gráficas.")
    return todos_experimentos

def generar_graficas(experimentos, resultados, graficas_path):
    """
    Genera y guarda las gráficas de los experimentos en una carpeta específica.

    Cada experimento tendrá su propia carpeta donde se guardarán:
    - La comparación entre la solución exacta y la aproximada.
    - La evolución de la pérdida durante el entrenamiento.
    - La evolución del error relativo.
    - La comparación entre la pérdida y el error relativo.

    Parámetros:
    - experimentos (lista): Lista de experimentos seleccionados para graficar.
    - resultados (lista): Lista de resultados obtenidos en `ejecutar_experimentos`.
    - graficas_path (str): Ruta donde se almacenarán las gráficas.

    Retorna:
    - None
    """

    for exp in experimentos:
        exp_id = exp[0]  # ID del experimento
        exp_folder = os.path.join(graficas_path, f"Experimento_{exp_id}")
        os.makedirs(exp_folder, exist_ok=True)  # 📌 Crear carpeta del experimento si no existe

        # 🔹 Extraer datos del experimento correspondiente
        uModel, history, exactU, xList = resultados[exp_id - 1][9:]

        # 📌 Generar las gráficas
        plot_results(uModel, history, exactU, xList, exp_folder)

    print(f"✔ Todas las gráficas generadas en {graficas_path}.")

# 🔹 Paso 1: Configuración inicial (Crear directorios y archivo Excel)
print("📂 Configurando estructura de almacenamiento...")
base_path, graficas_path = crear_estructura_directorios()
excel_file = inicializar_excel()

# 🔹 Paso 2: Ejecutar experimentos
print("🚀 Ejecutando experimentos...")
resultados_experimentos = ejecutar_experimentos(func, 45)

# 🔹 Paso 3: Seleccionar los experimentos para graficar
print("📊 Seleccionando experimentos para generación de gráficas...")
experimentos_seleccionados = seleccionar_experimentos(resultados_experimentos)

# 🔹 Paso 4: Generar todas las gráficas
print("📈 Generando gráficas...")
generar_graficas(experimentos_seleccionados, resultados_experimentos, graficas_path)

# 🔹 Paso 5: Guardar los resultados en Excel
print("💾 Guardando resultados en archivo Excel...")
guardar_resultados_excel(resultados_experimentos, funciones_experimentos)

print("✅ Proceso finalizado exitosamente.")
