import coremltools as ct
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_onnx_to_coreml(onnx_path="yolov8n.onnx"):
    try:
        logger.info(f"Loading ONNX model from {onnx_path}")
        
        # Convert with source specified as pytorch
        model_coreml = ct.convert(
            onnx_path,
            source='pytorch',  # Specify the source framework
            inputs=[
                ct.ImageType(
                    name="images",
                    shape=(1, 3, 640, 640),
                    color_layout="BGR",
                    scale=1/255.0,
                )
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16
        )
        
        logger.info("Saving Core ML model")
        model_coreml.save("YOLOv8.mlmodel")
        
        logger.info("Conversion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        convert_onnx_to_coreml()
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")