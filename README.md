# Video Upscaling con Real-ESRGAN y ONNX Runtime

---

## Resumen

Este proyecto implementa un sistema de mejora de resolución para video, utilizando el modelo **Real-ESRGAN** en formato ONNX y la biblioteca OpenCV en C++.  
El programa toma un video, aplica super resolución a cada fotograma y muestra el resultado en pantalla, permitiendo aprovechar tanto CPU como GPU (CUDA) según la configuración.

---

## Funcionalidades

- Lectura y procesamiento de video con OpenCV.
- Inferencia de super resolución usando Real-ESRGAN (ONNX).
- Selección dinámica de dispositivo de ejecución (CPU o GPU).
- Visualización en tiempo real con indicador de FPS.

---

## Dependencias

- C++17 o superior
- OpenCV (con soporte para video)
- ONNX Runtime (CPU o GPU)
- CUDA (opcional, solo si se desea usar GPU)
- Linux (desarrollado y probado en Ubuntu)

---

## Organización del repositorio

```
├── models/
│   └── RealESRGAN_x4.onnx   # Modelo ONNX preentrenado
├── video5.mp4               # Video de ejemplo
├── principal.cpp            # Código fuente principal
├── CMakeLists.txt           # Configuración de compilación
├── README.md                # Documentación (este archivo)
```

---

## Instrucciones de uso

1. Clona este repositorio.
2. Instala las dependencias necesarias (OpenCV, ONNX Runtime).
3. Verifica y ajusta las rutas del modelo y video en `principal.cpp` si es necesario.
4. Compila el proyecto usando CMake:
   ```sh
   cmake -DONNXRUNTIME_DIR=/ruta/a/onnxruntime .
   make
   ```
5. Ejecuta el programa:
   ```sh
   ./principal
   ```
   Puedes modificar la variable para seleccionar CPU o GPU en el código fuente.

---

## Notas y recomendaciones

- El uso de GPU (CUDA) acelera notablemente el procesamiento, permitiendo resultados en tiempo real.
- Si tu equipo no dispone de GPU compatible, el programa funcionará en CPU, aunque a menor velocidad.
- Para videos de alta resolución o equipos con poca memoria, puedes reducir el tamaño de entrada modificando la variable de escala en el código.

---

## Créditos

Trabajo académico para la materia de Visión por Computador.  
Implementación y adaptación por [Tu Nombre].

---

