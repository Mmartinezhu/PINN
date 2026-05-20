import tensorflow as tf  # Librería para aprendizaje profundo
import numpy as np  # Librería para manipulación de matrices y cálculos numéricos
import matplotlib.pyplot as plt  # Librería para visualización de gráficos
from matplotlib import rcParams  # Configuración de gráficos
import os  # Manejo de archivos y directorios
import pandas as pd  # Manejo de datos en formato tabular
os.environ["KERAS_BACKEND"] = "tensorflow"  # Define TensorFlow como backend de Keras
import keras_core as keras  # Librería de redes neuronales basada en TensorFlow
import gc  # Para gestión de la memoria (recolección de basura)
import psutil  # Para medir uso de CPU durante la ejecución de experimentos
import time

# Configuración de reproducibilidad
keras.utils.set_random_seed(1234)  # Fija una semilla para asegurar resultados reproducibles
dtype = 'float32'  # Establece la precisión de los cálculos
keras.backend.set_floatx(dtype)  # Aplica el tipo de dato definido

# Definición de parámetros globales
alpha = 1  # Parámetro utilizado en las ecuaciones diferenciales
limInf = 0  # Límite inferior del dominio de la solución
limSup = np.pi  # Límite superior del dominio de la solución
iterations = 400 # epocas por experimento

def makeModel2(neurons, nLayers, activation, limInf=0, limSup=np.pi):
    """
    Crea el modelo de la PINN con condiciones de frontera impuestas.
    
    Parámetros:
    - neurons (int): Número de neuronas por capa.
    - nLayers (int): Número de capas ocultas.
    - activation (str): Función de activación.
    - limInf (float): Límite inferior del dominio.
    - limSup (float): Límite superior del dominio.

    Retorna:
    - uModel (keras.Model): Modelo de la PINN con condiciones de frontera incorporadas.
    """
    xVals = keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)
    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(xVals)
    
    for _ in range(nLayers - 2):
        l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype)(l1)
    
    output = keras.layers.Dense(1, activation=activation, dtype=dtype)(l1)
    
    # 🔹 Impone las condiciones de frontera directamente en la salida
    boundaryC = keras.layers.Lambda(lambda x: (x - limInf) * (x - limSup))(xVals)
    output = keras.layers.Multiply()([output, boundaryC])
    
    uModel = keras.Model(inputs=xVals, outputs=output, name='u_model')
    return uModel
    
class Loss2(keras.layers.Layer):
    """
    Capa personalizada para calcular la pérdida en el Planteamiento 2.
    No penaliza las condiciones de frontera, ya que están impuestas en la arquitectura.
    """

    def __init__(self, uModel, nPts, f, limInf=0, limSup=np.pi, **kwargs):
        super(Loss2, self).__init__()
        self.uModel = uModel
        self.nPts = nPts
        self.f = f
        self.limInf = limInf
        self.limSup = limSup

    def call(self, inputs):
        x = tf.random.uniform([self.nPts], dtype=dtype, minval=self.limInf, maxval=self.limSup)
        
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                u = self.uModel(x, training=True)
            dux = t2.gradient(u, x)
        duxx = t1.gradient(dux, x)

        errorPDE = keras.ops.mean((-duxx + alpha * u - self.f(x)) ** 2)
        return errorPDE

def makeLossModel2(uModel, nPts, f, limInf=0, limSup=np.pi):
    """
    Crea el modelo de pérdida basado en la ecuación diferencial para el Planteamiento 2.

    Parámetros:
    - uModel (keras.Model): Modelo de la PINN.
    - nPts (int): Número de puntos de muestreo en el dominio.
    - f (función): Función de referencia f(x).
    - limInf (float): Límite inferior del dominio.
    - limSup (float): Límite superior del dominio.

    Retorna:
    - lossModel (keras.Model): Modelo de pérdida para la PINN.
    """

    # 🔹 Entrada del modelo (valores de x en el dominio)
    xVals = keras.layers.Input(shape=(1,), name='x_input', dtype=dtype)

    # 🔹 Capa de pérdida personalizada
    loss_output = Loss2(uModel, nPts, f, limInf, limSup)(xVals)

    # 🔹 Creación del modelo de pérdida
    lossModel = keras.Model(inputs=xVals, outputs=loss_output)

    return lossModel

class RelativeErrorCallback(tf.keras.callbacks.Callback):
    """
    Callback personalizado para calcular el error relativo en norma H¹ al final de cada época.
    """

    def __init__(self, uModel, exactU, nPts):
        super().__init__()
        self.uModel = uModel
        self.exactU = exactU
        self.nPts = nPts

    def on_epoch_end(self, epoch, logs=None):
        Sval = tf.experimental.numpy.linspace(0., np.pi, num=self.nPts * 10, dtype=dtype)

        with tf.GradientTape(persistent=True) as t1:
            t1.watch(Sval)
            ueval_val = self.uModel(Sval)  # Solución aproximada de la PINN
            u_x_val = t1.gradient(ueval_val, Sval)  # Derivada de la solución aproximada
            u_e = self.exactU(Sval)  # Solución exacta
            ue_x = t1.gradient(u_e, Sval)  # Derivada de la solución exacta
        del t1  # Liberar memoria

        errorH01 = tf.reduce_mean((ue_x - u_x_val) ** 2)
        norm_exact = tf.reduce_mean(ue_x ** 2)
        relative_error = errorH01 / norm_exact

        logs['relative_error'] = relative_error

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

def train_PINN(hiperparametros, funcion_referencia, funcion_exacta):
    """
    Entrena la PINN con los hiperparámetros dados y la función de referencia seleccionada.

    Parámetros:
    - hiperparametros (dict): Diccionario con la configuración de la red (neurons, nLayers, etc.).
    - funcion_referencia (función): Función f(x) en la ecuación diferencial que la PINN debe aprender.
    - funcion_exacta (función): Solución exacta esperada para evaluar el error relativo.

    Retorna:
    - error_relativo (float): Error relativo entre la solución de la PINN y la solución exacta.
    - cpu_cost (float): Tiempo total de CPU consumido durante el entrenamiento.
    - history (tf.keras.callbacks.History): Historial del entrenamiento.
    - uModel (keras.Model): Modelo entrenado de la PINN.
    - xList (array): Puntos en el dominio donde se evaluó la solución.
    """

    # 🔹 Medir tiempo de CPU antes del entrenamiento
    process = psutil.Process(os.getpid())
    cpu_times_before = process.cpu_times()

    # 🔹 Construcción del modelo de red neuronal PINN
    uModel = makeModel2(
        neurons=hiperparametros["neurons"],
        nLayers=hiperparametros["nLayers"],
        activation='tanh',
        limInf=limInf,
        limSup=limSup
    )

    # 🔹 Definir el modelo de pérdida basado en la ecuación diferencial
    lossModel = makeLossModel2(
        uModel,
        hiperparametros["nPts"],
        funcion_referencia,
        limInf, limSup
    )

    # 🔹 Configurar el optimizador y la función de pérdida
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    lossModel.compile(optimizer=optimizer, loss=trickyLoss)

    relative_error_callback = RelativeErrorCallback(uModel, funcion_exacta, hiperparametros["nPts"])

    # 🔹 Entrenamiento de la PINN
    history = lossModel.fit(
        np.array([1.]), np.array([1.]), epochs=iterations, verbose=0, callbacks=[relative_error_callback]
    )

    # 🔹 Evaluar la solución aproximada y compararla con la solución exacta
    error_relativo = history.history.get('relative_error', [np.nan])[-1]

    # 🔹 Medir tiempo de CPU después del entrenamiento
    cpu_times_after = process.cpu_times()

    # 🔹 Calcular tiempo total de CPU usado (sumando usuario y sistema)
    # 🔹 Medir el porcentaje de CPU utilizado durante el entrenamiento
    cpu_cost, _ = medir_costo_computacional(
        lossModel.fit, np.array([1.]), np.array([1.]), epochs=iterations, verbose=0, callbacks=[relative_error_callback]
    )

    # 🔹 Generar puntos en el dominio para evaluar la solución entrenada
    xList = np.linspace(limInf, limSup, 1000)

    # 🔹 Liberar memoria
    keras.backend.clear_session()
    gc.collect()

    return error_relativo, cpu_cost, history, uModel, xList

def ejecutar_experimentos(funciones_experimentos, n_experimentos=45):  
    """
    Ejecuta múltiples experimentos de entrenamiento de la PINN con los primeros hiperparámetros por defecto y los siguientes con valores predefinidos.

    Parámetros:
    - funciones_experimentos (lista): Lista de funciones de referencia y soluciones exactas.
    - n_experimentos (int, opcional): Número total de experimentos a ejecutar.

    Retorna:
    - resultados (lista): Lista de resultados de los experimentos, donde cada elemento contiene:
      [ID, neurons, nLayers, nPts, error_relativo, costo_computacional, modelo entrenado, historial, solución exacta, xList].
    """

    # 📌 Hiperparámetros fijos dados por los experimentos del planteamiento 1
    hiperparametros_fijos = [
        {"neurons": 7, "nLayers": 4, "nPts": 1164},
        {"neurons": 6, "nLayers": 4, "nPts": 1682},
        {"neurons": 4, "nLayers": 4, "nPts": 1469},
        {"neurons": 4, "nLayers": 5, "nPts": 1600},
        {"neurons": 6, "nLayers": 5, "nPts": 1245},
        {"neurons": 7, "nLayers": 4, "nPts": 2485},
        {"neurons": 5, "nLayers": 4, "nPts": 990},
        {"neurons": 4, "nLayers": 4, "nPts": 697},
        {"neurons": 8, "nLayers": 5, "nPts": 1608},
        {"neurons": 4, "nLayers": 4, "nPts": 1610},
        {"neurons": 9, "nLayers": 4, "nPts": 1889},
        {"neurons": 9, "nLayers": 4, "nPts": 2792},
        {"neurons": 8, "nLayers": 4, "nPts": 2630},
        {"neurons": 10, "nLayers": 5, "nPts": 2982}
    ]

    num_lotes = min(len(funciones_experimentos), n_experimentos)  
    lote_size = n_experimentos // num_lotes  

    resultados = []

    for i in range(num_lotes):
        f_rhs, exactU = funciones_experimentos[i]  
        print(f"\n🔹 Ejecutando Lote {i+1}...\n")

        for j in range(lote_size):
            experimento_id = i * lote_size + j + 1
            print(f"   ▶ Ejecutando experimento {experimento_id}...")

            # 📌 Usar los hiperparámetros por defecto para el primer experimento de cada lote
            if j == 0:
                hiperparametros = hiperparametros_por_defecto()
            else:
                # 📌 Usar los 14 hiperparámetros fijos en lugar de aleatorios
                hiperparametros = hiperparametros_fijos[(j - 1) % len(hiperparametros_fijos)]

            error_relativo, cpu_cost, history, uModel, xList = train_PINN(hiperparametros, f_rhs, exactU)

            print(f"🔹 Completado - Error relativo: {error_relativo:.5f}, Uso CPU: {cpu_cost:.2f}%")

            resultados.append([
                experimento_id, 
                hiperparametros["neurons"], 
                hiperparametros["nLayers"], 
                hiperparametros["nPts"], 
                error_relativo, 
                cpu_cost,  
                uModel,   
                history,  
                exactU,   
                xList     
            ])

    return resultados

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
    }

def crear_estructura_directorios(base_path="experimentos/PLANTEAMIENTO_2"):
    """
    Crea la estructura de directorios para almacenar los resultados de los experimentos.

    Parámetros:
    - base_path (str): Ruta base donde se almacenarán los resultados.

    Retorna:
    - base_path (str): Ruta base de los experimentos.
    - graficas_path (str): Ruta donde se guardarán las gráficas generadas.
    """
    os.makedirs(base_path, exist_ok=True)
    graficas_path = os.path.join(base_path, "graficas")
    os.makedirs(graficas_path, exist_ok=True)

    return base_path, graficas_path

def plot_results(uModel, history, exactU, xList, exp_folder):
    """
    Genera y guarda las gráficas de un experimento, asegurando que se use la función exacta correcta.

    Parámetros:
    - uModel (keras.Model): Modelo entrenado de la PINN.
    - history (tf.keras.callbacks.History): Historial de entrenamiento.
    - exactU (función): Función de la solución exacta.
    - xList (array): Puntos en el dominio donde se evaluó la solución.
    - exp_folder (str): Ruta donde se guardarán las gráficas.

    Retorna:
    - None
    """

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 14
    rcParams['legend.fontsize'] = 12
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['axes.labelsize'] = 14

    # 🔹 Comparación de la solución exacta y la aproximada
    fig, ax = plt.subplots()
    plt.plot(xList, uModel(xList), color='b', label="u_approx")
    plt.plot(xList, exactU(xList), color='m', label="u_exact")  
    plt.legend()
    ax.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "comparacion_u.png"), dpi=300)
    plt.close()

    # 🔹 Evolución de la pérdida y error relativo
    fig, ax1 = plt.subplots()
    ax1.plot(history.history['loss'], color='g', label="Loss")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='g')

    # 🔹 Verificar que 'relative_error' existe en history.history antes de graficar
    if 'relative_error' in history.history:
        ax2 = ax1.twinx()
        ax2.plot(history.history['relative_error'], color='b', label="Relative Error")
        ax2.set_ylabel("Relative Error", color='b')

    ax1.grid(which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_folder, "error_vs_loss.png"), dpi=300)
    plt.close()

    # 🔹 Error relativo a lo largo de las épocas
    if 'relative_error' in history.history:
        fig, ax = plt.subplots()
        plt.plot(history.history['relative_error'], color='b', label="Relative Error")
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
    num_lotes = min(len(func), len(resultados)) if len(resultados) > 0 else 1
    lote_size = max(1, len(resultados) // num_lotes) if num_lotes > 0 else 1

    # 📌 Lista para almacenar los experimentos seleccionados
    todos_experimentos = []

    for i in range(num_lotes):
        # 🔹 Seleccionar experimentos de este lote
        lote_resultados = [r for r in resultados if (r[0] - 1) // lote_size == i]
        todos_experimentos.extend(lote_resultados)

    print("✔ Experimentos listos para generar gráficas.")
    return todos_experimentos

def generar_graficas(experimentos, resultados, graficas_path):
    for exp in experimentos:
        exp_id = exp[0]  # ID del experimento
        exp_folder = os.path.join(graficas_path, f"Experimento_{exp_id}")
        os.makedirs(exp_folder, exist_ok=True)

        # Verificar que el resultado tiene 10 elementos
        if len(resultados[exp_id - 1]) < 10:
            print(f"⚠ Error: Resultados del experimento {exp_id} tienen solo {len(resultados[exp_id - 1])} elementos.")
            continue  # Saltar este experimento

        try:
            # Acceder a los valores correctos en lugar de hacer slicing [9:]
            uModel = resultados[exp_id - 1][6]
            history = resultados[exp_id - 1][7]
            exactU = resultados[exp_id - 1][8]
            xList = resultados[exp_id - 1][9]

            # Generar gráficas
            plot_results(uModel, history, exactU, xList, exp_folder)

        except Exception as e:
            print(f"⚠ Error en la generación de gráficas para el experimento {exp_id}: {e}")
            continue

    print(f"✔ Todas las gráficas generadas en {graficas_path}.")

def guardar_resultados_excel(resultados_experimentos, funciones_experimentos, carpeta="experimentos/PLANTEAMIENTO_2", archivo="resultados.xlsx"):
    """
    Guarda los resultados en un archivo Excel con hojas separadas para cada función de referencia.

    Parámetros:
    - resultados_experimentos (lista): Lista de resultados de los experimentos.
    - funciones_experimentos (lista): Lista de funciones de referencia utilizadas.
    - carpeta (str): Ruta de la carpeta donde se guardará el archivo Excel.
    - archivo (str): Nombre del archivo Excel.

    Retorna:
    - None
    """

    # 📌 Asegurar que la carpeta de destino existe
    os.makedirs(carpeta, exist_ok=True)
    
    # 📌 Ruta completa del archivo
    ruta_excel = os.path.join(carpeta, archivo)

    # 📌 Definir las columnas que queremos conservar (eliminando 'cpu_percent')
    columnas_base = ["ID", "neurons", "nLayers", "nPts", "error_relativo", "cpu_cost"]

    # 📌 Crear un diccionario para agrupar resultados por función de referencia
    resultados_por_funcion = {func["title"]: [] for func in funciones_experimentos}

    # 📌 Organizar los resultados en función de la función de referencia utilizada
    num_experimentos_por_funcion = len(resultados_experimentos) // len(funciones_experimentos)

    for i, resultado in enumerate(resultados_experimentos):
        funcion_index = i // num_experimentos_por_funcion
        if funcion_index >= len(funciones_experimentos):
            continue  # Evitar errores si hay más resultados que funciones
        
        titulo_funcion = funciones_experimentos[funcion_index]["title"]

        # 📌 Convertir valores a float si son tensores o arrays
        fila_convertida = [
            float(val.numpy()) if isinstance(val, tf.Tensor) else 
            float(val) if isinstance(val, np.ndarray) else val 
            for val in resultado[:6]  
        ]

        resultados_por_funcion[titulo_funcion].append(fila_convertida)

    # 📌 Guardar en un archivo Excel con múltiples hojas
    with pd.ExcelWriter(ruta_excel, engine='xlsxwriter') as writer:
        for titulo, datos in resultados_por_funcion.items():
            # 🔹 Limpiar caracteres no permitidos en nombres de hoja
            titulo_limpio = titulo.replace("*", "x").replace("/", "-").replace("\\", "-").replace("?", "").replace("[", "").replace("]", "").replace(":", "")
            titulo_limpio = titulo_limpio[:31]  # Excel no permite más de 31 caracteres

            df = pd.DataFrame(datos, columns=columnas_base)
            df.to_excel(writer, sheet_name=titulo_limpio, index=False)

    print(f"✔ Resultados guardados en {ruta_excel} con hojas separadas para cada función de referencia.")

def flujo_experimentos_planteamiento_2():
    """
    Ejecuta el sistema completo de experimentos para el Planteamiento 2, incluyendo:
    - Creación de directorios.
    - Ejecución de experimentos con múltiples configuraciones.
    - Almacenamiento de resultados en Excel.
    - Generación de gráficas.
    """
    print("📂 Configurando estructura de almacenamiento...")
    base_path, graficas_path = crear_estructura_directorios()

    print("🚀 Ejecutando experimentos...")
    resultados_experimentos = ejecutar_experimentos(func, 45)

    print("📊 Seleccionando experimentos para generación de gráficas...")
    experimentos_seleccionados = seleccionar_experimentos(resultados_experimentos)

    print("📈 Generando gráficas...")
    generar_graficas(experimentos_seleccionados, resultados_experimentos, graficas_path)

    print("💾 Guardando resultados en archivo Excel...")
    guardar_resultados_excel(resultados_experimentos, funciones_experimentos=funciones_experimentos)

    print("✅ Proceso finalizado exitosamente.")

flujo_experimentos_planteamiento_2()
