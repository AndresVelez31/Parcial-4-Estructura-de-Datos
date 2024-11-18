import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Función de merge sort
def merge_sort(arr):
    # Diccionario para almacenar los contadores de comparaciones e intercambios
    contadores = {'comparaciones': 0, 'intercambios': 0}

    def _merge_sort(arr):
        if len(arr) > 1:  # Si el arreglo tiene más de un elemento, lo dividimos
            mitad = len(arr) // 2  # Encuentra el punto medio
            mitad_izquierda = arr[:mitad]  # Parte izquierda del arreglo
            mitad_derecha = arr[mitad:]  # Parte derecha del arreglo

            # Llamadas recursivas para ordenar ambas mitades
            _merge_sort(mitad_izquierda)
            _merge_sort(mitad_derecha)

            # Inicializamos los índices para recorrer las dos mitades
            i = j = k = 0

            # Compara los elementos de las dos mitades y los ordena en el arreglo original
            while i < len(mitad_izquierda) and j < len(mitad_derecha):
                contadores['comparaciones'] += 1  
                
                if mitad_izquierda[i] < mitad_derecha[j]:
                    arr[k] = mitad_izquierda[i]  # Toma el elemento de la mitad izquierda
                    i += 1
                else:
                    arr[k] = mitad_derecha[j]  # Toma el elemento de la mitad derecha
                    j += 1
                k += 1
                contadores['intercambios'] += 1  

            # Si quedan elementos en la mitad izquierda, los coloca en el arreglo
            while i < len(mitad_izquierda):
                arr[k] = mitad_izquierda[i]
                i += 1
                k += 1
                contadores['intercambios'] += 1  

            # Si quedan elementos en la mitad derecha, los coloca en el arreglo
            while j < len(mitad_derecha):
                arr[k] = mitad_derecha[j]
                j += 1
                k += 1
                contadores['intercambios'] += 1  

    # Llamada inicial a la función recursiva
    _merge_sort(arr)
    return contadores['comparaciones'], contadores['intercambios'] 


# Función de selection sort con contadores de comparaciones e intercambios
def selection_sort(arr):
    comparaciones = 0  # Contador de comparaciones
    intercambios = 0   # Contador de intercambios

    # Recorre todo el arreglo
    for i in range(len(arr)):
        act_idx_min = i  # El índice del elemento más pequeño en la parte no ordenada del arreglo
        
        # Compara el elemento actual con los elementos siguientes
        for j in range(i + 1, len(arr)):
            comparaciones += 1  # Se realiza una comparación
            if arr[j] < arr[act_idx_min]:
                act_idx_min = j  # Si encuentra un elemento menor, actualiza el índice del mínimo

        # Si el índice del mínimo no coincide con el índice actual, intercambia los elementos
        if act_idx_min != i:
            arr[i], arr[act_idx_min] = arr[act_idx_min], arr[i]
            intercambios += 1  # Se realiza un intercambio

    return comparaciones, intercambios  # Devuelve los contadores de comparaciones e intercambios


# Función de heap sort con contadores de comparaciones e intercambios
def heap(arr):
    comparaciones = 0  # Contador de comparaciones
    intercambios = 0   # Contador de intercambios
    
    # Construcción del montículo máximo
    for i in range(len(arr) // 2 - 1, -1, -1):
        comparaciones_locales, intercambios_locales = maxHeap(arr, len(arr), i)
        comparaciones += comparaciones_locales
        intercambios += intercambios_locales
    
    # Ordenamiento del montículo
    for i in range(len(arr) - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Intercambia el más grande con el último desordenado
        intercambios += 1
        comparaciones_locales, intercambios_locales = maxHeap(arr, i, 0)
        comparaciones += comparaciones_locales
        intercambios += intercambios_locales
    
    return comparaciones, intercambios


def maxHeap(arr, n, padre):
    masGrande = padre  # Inicializa el mayor como el padre
    izq = (2 * padre) + 1  # Índice del hijo izquierdo
    der = (2 * padre) + 2  # Índice del hijo derecho
    
    comparaciones = 0  # Contador de comparaciones locales
    intercambios = 0   # Contador de intercambios locales
    
    # Compara el hijo izquierdo con el padre
    if izq < n:
        comparaciones += 1
        if arr[izq] > arr[masGrande]:
            masGrande = izq

    # Compara el hijo derecho con el mayor actual
    if der < n:
        comparaciones += 1
        if arr[der] > arr[masGrande]:
            masGrande = der
    
    # Si el mayor no es el padre, intercambia
    if masGrande != padre:
        arr[masGrande], arr[padre] = arr[padre], arr[masGrande]
        intercambios += 1
        # Ajusta el subárbol afectado de forma recursiva
        comparaciones_recursivas, intercambios_recursivos = maxHeap(arr, n, masGrande)
        comparaciones += comparaciones_recursivas
        intercambios += intercambios_recursivos
    
    return comparaciones, intercambios


# Función de shell sort con contadores de comparaciones e intercambios
def shell(arr):
    n = len(arr)
    intervalo = n // 2
    comparaciones = 0  # Contador de comparaciones
    intercambios = 0   # Contador de intercambios

    while intervalo > 0:
        for i in range(intervalo, n):     
            j = i
            aux = arr[i]

            # Comparar e intercambiar según el intervalo
            while j >= intervalo:
                comparaciones += 1  # Se realiza una comparación
                if arr[j - intervalo] > aux:
                    arr[j] = arr[j - intervalo]
                    j -= intervalo
                    intercambios += 1
                else:
                    break
            arr[j] = aux
        intervalo //= 2

    return comparaciones, intercambios


# Función de insertion sort con contadores de comparaciones e intercambios
def insertion(lista):
    comparaciones = 0  # Contador de comparaciones
    intercambios = 0   # Contador de intercambios

    for i in range(len(lista)):
        actual = lista[i]  # Guarda el valor del elemento actual en una variable temporal
        j = i

        # Mientras el elemento no sea el primero y el elemento anterior sea mayor que este, intercambiar
        while j > 0:
            comparaciones += 1  # Se realiza una comparación
            if lista[j - 1] > actual:
                lista[j] = lista[j - 1]  # Desplaza el elemento anterior una posición adelante
                intercambios += 1  # Se realiza un intercambio
                j -= 1
            else:
                break  # Si no es necesario intercambiar, salir del bucle
        
        lista[j] = actual  # Una vez encuentra la posición correcta, actualiza el valor
    
    return comparaciones, intercambios


# Función de bubble sort con contadores de comparaciones e intercambios
def bubble(arreglo):
    n = len(arreglo)
    comparaciones = 0
    intercambios = 0
    
    for i in range(1, n):        
        for j in range(0, n - i):  # Optimización: No revisar las posiciones ya ordenadas
            comparaciones += 1
            if arreglo[j] > arreglo[j + 1]:
                aux = arreglo[j]
                arreglo[j] = arreglo[j + 1]
                arreglo[j + 1] = aux
                intercambios += 1
    return comparaciones, intercambios


# Función para generar listas de prueba
def generar_lista(tamaño, tipo='aleatorio'):
    if tipo == 'aleatorio':
        return [random.randint(1, 1000) for _ in range(tamaño)]
    elif tipo == 'ordenado':
        return list(range(tamaño))
    elif tipo == 'reverso':
        return list(range(tamaño, 0, -1))
    elif tipo == 'medianamente_ordenada':
        # Lista ordenada parcialmente desordenada
        lista = list(range(tamaño))
        for _ in range(tamaño // 3):  # Desordenamos una fracción de la lista
            i, j = random.sample(range(tamaño), 2)
            lista[i], lista[j] = lista[j], lista[i]
        return lista


# Medición del tiempo de ejecución
def medir_tiempo(func, arr):
    start_time = time.time()
    comparaciones, intercambios = func(arr)
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
    return tiempo_ejecucion, comparaciones, intercambios


# Tamaños de las listas de prueba
tamaños = [100, 5000, 10000, 15000, 25000]

# Tipos de listas de prueba
tipos_lista = ['aleatorio', 'ordenado', 'reverso', 'medianamente_ordenada']

# Diccionarios para almacenar los tiempos de ejecución
tiempos_merge = {tipo: [] for tipo in tipos_lista}
tiempos_selection = {tipo: [] for tipo in tipos_lista}
tiempos_heap = {tipo: [] for tipo in tipos_lista}
tiempos_shell = {tipo: [] for tipo in tipos_lista}
tiempos_insertion = {tipo: [] for tipo in tipos_lista}
tiempos_bubble = {tipo: [] for tipo in tipos_lista}

# Almacenar las comparaciones de cada algoritmo para cada tipo de lista
comparaciones_merge = {tipo: [] for tipo in tipos_lista}
comparaciones_selection = {tipo: [] for tipo in tipos_lista}
comparaciones_heap = {tipo: [] for tipo in tipos_lista}
comparaciones_shell = {tipo: [] for tipo in tipos_lista}
comparaciones_insertion = {tipo: [] for tipo in tipos_lista}
comparaciones_bubble = {tipo: [] for tipo in tipos_lista}

# Almacenar los intercambios de cada algoritmo para cada tipo de lista
intercambios_merge = {tipo: [] for tipo in tipos_lista}
intercambios_selection = {tipo: [] for tipo in tipos_lista}
intercambios_heap = {tipo: [] for tipo in tipos_lista}
intercambios_shell = {tipo: [] for tipo in tipos_lista}
intercambios_insertion = {tipo: [] for tipo in tipos_lista}
intercambios_bubble = {tipo: [] for tipo in tipos_lista}

# Diccionario para almacenar los resultados por tipo de lista
resultados = {tipo: [] for tipo in tipos_lista}

# Medir los tiempos para cada algoritmo y tipo de lista
for tipo in tipos_lista:
    for tamaño in tamaños:
        lista = generar_lista(tamaño, tipo)
        
        # Merge Sort
        tiempo_merge, comparaciones_merge_val, intercambios_merge_val = medir_tiempo(merge_sort, lista.copy())
        tiempos_merge[tipo].append(tiempo_merge)
        comparaciones_merge[tipo].append(comparaciones_merge_val)
        intercambios_merge[tipo].append(intercambios_merge_val)
        resultados[tipo].append([tamaño, "Merge Sort", tiempo_merge, comparaciones_merge_val, intercambios_merge_val])

        # Selection Sort
        tiempo_selection, comparaciones_selection_val, intercambios_selection_val = medir_tiempo(selection_sort, lista.copy())
        tiempos_selection[tipo].append(tiempo_selection)
        comparaciones_selection[tipo].append(comparaciones_selection_val)
        intercambios_selection[tipo].append(intercambios_selection_val)
        resultados[tipo].append([tamaño, "Selection Sort", tiempo_selection, comparaciones_selection_val, intercambios_selection_val])

        # Heap Sort
        tiempo_heap, comparaciones_heap_val, intercambios_heap_val = medir_tiempo(heap, lista.copy())
        tiempos_heap[tipo].append(tiempo_heap)
        comparaciones_heap[tipo].append(comparaciones_heap_val)
        intercambios_heap[tipo].append(intercambios_heap_val)
        resultados[tipo].append([tamaño, "Heap Sort", tiempo_heap, comparaciones_heap_val, intercambios_heap_val])

        # Shell Sort
        tiempo_shell, comparaciones_shell_val, intercambios_shell_val = medir_tiempo(shell, lista.copy())
        tiempos_shell[tipo].append(tiempo_shell)
        comparaciones_shell[tipo].append(comparaciones_shell_val)
        intercambios_shell[tipo].append(intercambios_shell_val)
        resultados[tipo].append([tamaño, "Shell Sort", tiempo_shell, comparaciones_shell_val, intercambios_shell_val])
        
        # Insertion Sort
        tiempo_insertion, comparaciones_insertion_val, intercambios_insertion_val = medir_tiempo(insertion, lista.copy())
        tiempos_insertion[tipo].append(tiempo_insertion)
        comparaciones_insertion[tipo].append(comparaciones_insertion_val)
        intercambios_insertion[tipo].append(intercambios_insertion_val)
        resultados[tipo].append([tamaño, "Insertion Sort", tiempo_insertion, comparaciones_insertion_val, intercambios_insertion_val])

        # Bubble Sort
        tiempo_bubble, comparaciones_bubble_val, intercambios_bubble_val = medir_tiempo(bubble, lista.copy())
        tiempos_bubble[tipo].append(tiempo_bubble)
        comparaciones_bubble[tipo].append(comparaciones_bubble_val)
        intercambios_bubble[tipo].append(intercambios_bubble_val)
        resultados[tipo].append([tamaño, "Bubble Sort", tiempo_bubble, comparaciones_bubble_val, intercambios_bubble_val])

# Graficar tiempos de ejecución
def graficar_tiempos(tiempos_merge, tiempos_selection, tiempos_heap, tiempos_shell, tiempos_insertion, tiempos_bubble, tamaños, tipo_lista, ax):
    ax.plot(tamaños, tiempos_merge, label=f"Merge Sort ({tipo_lista})", color='blue', marker='o', linestyle='-')
    ax.plot(tamaños, tiempos_selection, label=f"Selection Sort ({tipo_lista})", color='red', marker='o', linestyle='--')
    ax.plot(tamaños, tiempos_heap, label=f"Heap Sort ({tipo_lista})", color='green', marker='o', linestyle='-.')
    ax.plot(tamaños, tiempos_shell, label=f"Shell Sort ({tipo_lista})", color='purple', marker='o', linestyle=':')
    ax.plot(tamaños, tiempos_insertion, label=f"Insertion Sort ({tipo_lista})", color='orange', marker='o', linestyle='-.')
    ax.plot(tamaños, tiempos_bubble, label=f"Bubble Sort ({tipo_lista})", color='brown', marker='o', linestyle='-')

    ax.set_title(f"Tiempo de Ejecución de Ordenación ({tipo_lista.capitalize()})", fontsize=12)
    ax.set_xlabel("Tamaño de Entrada", fontsize=10)
    ax.set_ylabel("Tiempo (segundos)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

    # Escala logarítmica solo en el eje Y
    ax.set_yscale('log')

# Graficar resultados
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Graficar los tiempos para cada tipo de lista
for i, tipo in enumerate(tipos_lista):
    ax = axs[i // 2, i % 2]
    graficar_tiempos(tiempos_merge[tipo], tiempos_selection[tipo], tiempos_heap[tipo], tiempos_shell[tipo], tiempos_insertion[tipo], tiempos_bubble[tipo], tamaños, tipo, ax)

plt.tight_layout()
plt.show()

# Graficar comparaciones
def graficar_comparaciones(comparaciones_merge, comparaciones_selection, comparaciones_heap, comparaciones_shell, comparaciones_insertion, comparaciones_bubble, tamaños, tipo_lista, ax):
    ax.plot(tamaños, comparaciones_merge, label=f"Merge Sort ({tipo_lista})", color='blue', marker='o', linestyle='--')
    ax.plot(tamaños, comparaciones_selection, label=f"Selection Sort ({tipo_lista})", color='red', marker='o', linestyle='--')
    ax.plot(tamaños, comparaciones_heap, label=f"Heap Sort ({tipo_lista})", color='green', marker='o', linestyle='--')
    ax.plot(tamaños, comparaciones_shell, label=f"Shell Sort ({tipo_lista})", color='purple', marker='o', linestyle='--')
    ax.plot(tamaños, comparaciones_insertion, label=f"Insertion Sort ({tipo_lista})", color='orange', marker='o', linestyle='--')
    ax.plot(tamaños, comparaciones_bubble, label=f"Bubble Sort ({tipo_lista})", color='brown', marker='o', linestyle='--')

    ax.set_title(f"Comparaciones de Ordenación ({tipo_lista.capitalize()})", fontsize=12)
    ax.set_xlabel("Tamaño de Entrada", fontsize=10)
    ax.set_ylabel("Número de Comparaciones", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

    # Escala logarítmica solo en el eje Y
    ax.set_yscale('log')

# Graficar las comparaciones de cada tipo de lista
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Graficar las comparaciones para cada tipo de lista
for i, tipo in enumerate(tipos_lista):
    ax = axs[i // 2, i % 2]
    graficar_comparaciones(comparaciones_merge[tipo], comparaciones_selection[tipo], comparaciones_heap[tipo], comparaciones_shell[tipo], comparaciones_insertion[tipo], comparaciones_bubble[tipo], tamaños, tipo, ax)

plt.tight_layout()
plt.show()

# Función para mostrar los resultados en formato de tabla
def mostrar_tabla(resultados):
    # Creamos un DataFrame de pandas con los resultados
    df = pd.DataFrame(resultados, columns=["Tamaño", "Algoritmo", "Tiempo (s)", "Comparaciones", "Intercambios"])
    display(df)  # Usamos display para mostrar la tabla interactiva

# Mostrar las tablas para cada tipo de lista
for tipo in tipos_lista:
    print(f"\n--- Resultados con Lista {tipo.capitalize()} ---")
    mostrar_tabla(resultados[tipo])
