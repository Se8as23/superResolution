#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <chrono>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using namespace cv;

// Inicializa el entorno de ONNX Runtime
Ort::Env ortEnv(ORT_LOGGING_LEVEL_WARNING, "SuperResVideo");

// Preprocesa la imagen para el modelo
void prepararEntrada(const Mat& img, vector<float>& datos, vector<int64_t>& dims) {
    Mat imgRGB;
    cvtColor(img, imgRGB, COLOR_BGR2RGB);
    imgRGB.convertTo(imgRGB, CV_32F, 1.0 / 255.0);

    vector<Mat> canales(3);
    split(imgRGB, canales);

    int h = img.rows, w = img.cols;
    dims = {1, 3, h, w};
    datos.resize(3 * h * w);

    for (int c = 0; c < 3; ++c)
        memcpy(&datos[c * h * w], canales[c].data, h * w * sizeof(float));
}

// Postprocesa la salida del modelo a imagen OpenCV
Mat procesarSalida(float* salida, const vector<int64_t>& dims) {
    int canales = dims[1], alto = dims[2], ancho = dims[3];
    vector<Mat> mats(canales);

    for (int c = 0; c < canales; ++c)
        mats[c] = Mat(alto, ancho, CV_32F, salida + c * alto * ancho).clone();

    Mat imgRGB, imgBGR;
    merge(mats, imgRGB);
    imgRGB = imgRGB * 255.0f;
    imgRGB.convertTo(imgRGB, CV_8U);
    cvtColor(imgRGB, imgBGR, COLOR_RGB2BGR);
    return imgBGR;
}

// Ejecuta la super resolución sobre un frame
Mat aplicarSuperResolucion(const Mat& frame, Ort::Session& sesion, Ort::MemoryInfo& memInfo, 
                           const vector<const char*>& entradas, const vector<const char*>& salidas) {
    vector<float> datosEntrada;
    vector<int64_t> dimsEntrada;
    prepararEntrada(frame, datosEntrada, dimsEntrada);

    auto tensorEntrada = Ort::Value::CreateTensor<float>(
        memInfo, datosEntrada.data(), datosEntrada.size(), dimsEntrada.data(), dimsEntrada.size()
    );

    auto resultados = sesion.Run(Ort::RunOptions{nullptr}, entradas.data(), &tensorEntrada, 1, salidas.data(), 1);
    float* datosSalida = resultados[0].GetTensorMutableData<float>();
    auto dimsSalida = resultados[0].GetTensorTypeAndShapeInfo().GetShape();

    return procesarSalida(datosSalida, dimsSalida);
}

int main() {
    // Configuración de rutas
    string rutaModelo = "models/RealESRGAN_x2.onnx";
    string rutaVideo = "testvideo2.mp4";
    bool usarCUDA = false; // Cambia a true si quieres probar GPU

    // Configuración de sesión ONNX
    Ort::SessionOptions opciones;
    opciones.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opciones.EnableMemPattern();
    opciones.EnableCpuMemArena();

    if (usarCUDA) {
        if (OrtSessionOptionsAppendExecutionProvider_CUDA(opciones, 0) != nullptr) {
            cout << "No se pudo activar CUDA, usando CPU." << endl;
        } else {
            cout << "Procesando con GPU..." << endl;
        }
    } else {
        cout << "Procesando con CPU..." << endl;
    }

    Ort::Session sesion(ortEnv, rutaModelo.c_str(), opciones);
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Obtiene nombres de entrada y salida
    vector<const char*> nombresEntrada, nombresSalida;
    vector<Ort::AllocatedStringPtr> entradasPtrs, salidasPtrs;
    for (size_t i = 0; i < sesion.GetInputCount(); ++i) {
        entradasPtrs.emplace_back(sesion.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        nombresEntrada.push_back(entradasPtrs.back().get());
    }
    for (size_t i = 0; i < sesion.GetOutputCount(); ++i) {
        salidasPtrs.emplace_back(sesion.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
        nombresSalida.push_back(salidasPtrs.back().get());
    }

    // Abre el video
    VideoCapture video(rutaVideo);
    if (!video.isOpened()) {
        cout << "No se pudo abrir el video: " << rutaVideo << endl;
        return 1;
    }

    int ancho = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
    int alto = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
    float escala = 0.5f; // Cambia este valor para reducir aún más el tamaño de entrada
    int nuevoAncho = static_cast<int>(ancho * escala);
    int nuevoAlto = static_cast<int>(alto * escala);

    namedWindow("Super Resolución Video", WINDOW_NORMAL);

    int frames = 0;
    auto inicio = std::chrono::high_resolution_clock::now();

    Mat frame;
    while (video.read(frame)) {
        resize(frame, frame, Size(nuevoAncho, nuevoAlto));
        Mat frameSR = aplicarSuperResolucion(frame, sesion, memInfo, nombresEntrada, nombresSalida);

        frames++;
        auto ahora = std::chrono::high_resolution_clock::now();
        double segundos = std::chrono::duration<double>(ahora - inicio).count();
        double fps = frames / (segundos > 0 ? segundos : 1);

        putText(frameSR, "FPS: " + std::to_string(static_cast<int>(fps)), Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

        imshow("Super Resolución Video", frameSR);
        if (waitKey(1) == 27) break; // ESC para salir
    }

    video.release();
    destroyAllWindows();
    return 0;
}
