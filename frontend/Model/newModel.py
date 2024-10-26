import coremltools as ct
import onnx
from coremltools.models.neural_network import quantization_utils

def convert_to_coreml(onnx_model_path):
    # Load the ONNX model using the onnx package
    onnx_model = onnx.load(onnx_model_path)
    
    # Convert to Core ML format
    model_coreml = ct.convert(
        onnx_model,
        inputs=[
            ct.ImageType(
                name="images",
                shape=(1, 3, 640, 640),
                color_layout="BGR",
                scale=1/255.0
            )
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram"
    )
    
    # Save the model
    model_coreml.save("YOLOv8.mlmodel")
    
    # Optional quantization for model size/performance improvement
    model_coreml_quantized = quantization_utils.quantize_weights(model_coreml, nbits=8)
    model_coreml_quantized.save("YOLOv8_quantized.mlmodel")

    print("Models saved as YOLOv8.mlmodel and YOLOv8_quantized.mlmodel")
