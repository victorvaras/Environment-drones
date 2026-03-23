# 🚁 Entorno de simulación para redes moviles


Este repositorio contiene el entorno de simulación desarrollado para tesis de grado, enfocado en el despliegue de **Estaciones Base Aéreas (UAV-BS)** para proveer conectividad a usuarios con movilidad dinámica en entornos 5G.

## 🎬 Demo de la simulación

<img src="imagenes-readme/gif%20simulacion.gif" alt="GIF de simulación del entorno" width="900">

## 🎯 Objetivo del Proyecto
El sistema modela la interacción entre la dinámica de vuelo de un dron y metricas de calidad de servicio. El objetivo es disponer de un entorno para simulación para ser utilizado para entrenar trayectorias de vuelo a traves de  **Aprendizaje por Refuerzo (RL)**.

---

## 🔧 Componentes Clave

Este proyecto integra las siguientes tecnologías principales:

* **[Sionna](https://github.com/NVlabs/sionna)**: Simulación de capa física y trazado de rayos (Ray Tracing) para el entorno de red.
* **[PyFlyt](https://github.com/jjshoots/PyFlyt)**: Control y física de vuelo del UAV.
* **[Social Force](https://github.com/svenkreiss/socialforce)**: Modelado de movilidad de peatones basado en el modelo de Social Force.
* **[Gymnasium](https://github.com/farama-foundation/gymnasium)**: Interfaz estándar de agente-entorno para compatibilidad con RL.

---

## 🗂️ Estructura del Repositorio

- `Mapas-Sionna`: lugar donde se deben guardar mapas personalizados para simulación.
- `env/environment`: código fuente principal del proyecto.
- `env/pruebas_funcionamiento`: script para comprobar el sistema.
- `env/scripts`: scripts para ejecutar diferentes tipos de prueba del sistema.

---

## 💻 Requisitos del Sistema
Para garantizar la estabilidad de las simulaciones de trazado de rayos y la física del dron, se requiere:

* **Sistema Operativo:** Linux (Desarrollado y probado en **Ubuntu 22.04 LTS**).
* **Lenguaje:** Python 3.11.
* **Dependencias de Sistema:** `build-essential`, `python3.11-dev`.

---

## 🛠️ Guía de Instalación

Sigue estos pasos para configurar el entorno de desarrollo de manera local.

### 1. Preparación de Ubuntu 22.04
Instala Python 3.11 y las herramientas de compilación necesarias:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common build-essential python3-pip python3-venv
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils
```

### 2. Configuración del Proyecto y VENV
Clona este repositorio y crea un entorno virtual aislado:

```bash
# Ubícate en la carpeta donde deseas clonar el proyecto.

# Configurar entorno virtual (al mismo nivel que el repositorio)
python3.11 -m venv venv
source venv/bin/activate

# Clonar repositorio y entrar al proyecto
git clone https://github.com/victorvaras/Environment-drones.git
cd Environment-drones

# Asegurar que pip esté actualizado
pip install --upgrade pip
```

### 3. Instalación de Dependencias
Utiliza el archivo `requirements.txt` incluido para instalar todo el stack tecnológico:

```bash
# Ejecutar desde la raíz del repositorio (Environment-drones)
pip install -r requirements.txt
```

### 4. Prueba de Funcionamiento
Para validar una ejecución limpia del proyecto, usa el script base de prueba:

```bash
python env/pruebas_funcionamiento/run_prueba_funcionamiento.py
```

Al ejecutar el script se crea una carpeta con nombre similar a:

- `env/pruebas_funcionamiento/PRUEBA_FUNCIONAMIENTO_<timestamp>_...`

Dentro de esa carpeta deben generarse los siguientes archivos:

- `animacion_...gif`
- `PRx_dBm_3500MHz.png`
- `prueba_funcionamiento_metrics.csv`

Si estos archivos se generan correctamente, la instalación puede considerarse **exitosa**.

Si no se generan, o la ejecución falla, se recomienda:

1. Instalar dependencias faltantes con `pip install -r requirements.txt`.
2. Verificar que el entorno virtual esté activado.
3. Ponerse en contacto con los desarrolladores del proyecto para soporte.

### ⚠️ Solución de problemas con GPU (TensorFlow / CUDA)

**Advertencia:** Si al ejecutar la prueba aparece un error similar a:

```text
error: libdevice not found at ./libdevice.10.bc
tensorflow.python.framework.errors_impl.UnknownError:
JIT compilation failed. [Op:Pow]
```

o mensajes relacionados con GPU, CUDA, XLA o TensorFlow, esto indica que el entorno no logró utilizar correctamente la GPU.

**¿Qué significa?**

Este problema suele ocurrir cuando TensorFlow no encuentra o no puede cargar correctamente las librerías necesarias para trabajar con GPU.
Si deseas continuar ejecutando el proyecto sin usar GPU, puedes forzar la ejecución solo en CPU.

**Ejecutar temporalmente solo con CPU**

Usa el siguiente comando antes de ejecutar el script:

```bash
export CUDA_VISIBLE_DEVICES=""
```

Luego ejecuta normalmente la prueba:

```bash
python env/pruebas_funcionamiento/run_prueba_funcionamiento.py
```

**Ejecutar en una sola línea**

También puedes hacerlo directamente así:

```bash
CUDA_VISIBLE_DEVICES="" python env/pruebas_funcionamiento/run_prueba_funcionamiento.py
```

**Nota:** Al trabajar solo con CPU, el proyecto debería ejecutarse sin este error, aunque el rendimiento puede ser más lento en comparación con una ejecución con GPU correctamente configurada. Si se dispone de una GPU NVIDIA, se sugiere mantener la ejecución con GPU operativa para obtener mejor rendimiento.

---

## 📚 Documentación Complementaria

Para obtener información detallada sobre aspectos específicos del proyecto, consulta los siguientes documentos:

- **[Crear Mapas Personalizados](https://docs.google.com/document/d/1vflnjZuGj_a9-jnt0JIVjdgjpHt4OXL-NhI1ehBV4Kg/edit?usp=sharing)**: Guía completa para diseñar y crear mapas personalizados para el entorno de simulación.

- **[Modos de Vuelo del Dron](https://docs.google.com/document/d/1-vlN5dvGa8ktpp55Rz1sscl147YP9XGjL2-dLDvkFfI/edit?usp=sharing)**: Documentación sobre los diferentes modos de vuelo disponibles y cómo utilizarlos.

