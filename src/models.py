import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2, DenseNet121

# Make sure attention.py is in the same directory, or modify import if needed
try:
    from attention import cbam_block
except ImportError:
    from .attention import cbam_block

def build_custom_cnn_attention(input_shape=(224, 224, 3)):
    """
    Baseline Custom CNN with CBAM integrated.
    """
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x) # Integrate attention
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = cbam_block(x) # Integrate attention
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name="Custom_CNN_Attention")
    return model

def build_transfer_model(base_model_func, input_shape=(224, 224, 3), model_name="Transfer_Model"):
    """
    Generic function to build transfer learning models with CBAM.
    """
    base_model = base_model_func(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    
    # Inject CBAM attention after the convolutional base
    x = cbam_block(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(base_model.input, outputs, name=model_name)
    return model

def build_resnet50_attention(input_shape=(224, 224, 3)):
    return build_transfer_model(ResNet50, input_shape, "ResNet50_Attention")

def build_vgg16_attention(input_shape=(224, 224, 3)):
    return build_transfer_model(VGG16, input_shape, "VGG16_Attention")

def build_mobilenetv2_attention(input_shape=(224, 224, 3)):
    return build_transfer_model(MobileNetV2, input_shape, "MobileNetV2_Attention")

def build_densenet121_attention(input_shape=(224, 224, 3)):
    return build_transfer_model(DenseNet121, input_shape, "DenseNet121_Attention")
