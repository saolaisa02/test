 
        
from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import load_model
from src.keras_utils import load_model
import onnx
import keras2onnx

onnx_model_name = 'resnetcus.onnx'

# model = load_model('./final_models/IoU_05/utvm_models_resnet2_privatedata/utvm-model-privatedata-resnet2_backup_max')
model = load_model('final_models/IoU_05/utvm_models_resnet_cus_privatedata/utvm-model-privatedata-resnet-cus_backup_max')
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, onnx_model_name)
