import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import pandas as pd

def get_best_metric_from_csv(csv_path, metric='val_accuracy', mode='max'):
    if not csv_path or not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if metric in df.columns:
            if mode == 'max':
                return float(df[metric].max())
            else:
                return float(df[metric].min())
    except:
        pass
    return None

def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with Adam optimizer and binary crossentropy.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_model(model, train_data, val_data, epochs=20, model_path='best_model.h5', csv_log_path=None):
    """
    Train the model with callbacks for early stopping, checkpointing, and LR scheduling.
    """
    initial_acc = get_best_metric_from_csv(csv_log_path, 'val_accuracy', 'max')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.BackupAndRestore(backup_dir=f"backup_phase1_{model_path}")
    ]
    
    if csv_log_path:
        callbacks.append(tf.keras.callbacks.CSVLogger(csv_log_path, append=True))
        
    if initial_acc is not None:
        callbacks.append(ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1, initial_value_threshold=initial_acc))
    else:
        callbacks.append(ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1))
    
    print(f"Starting training for {model.name}...")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False
    )
    
    # GUARANTEE: Force Keras to load the mathematically perfect weights from disk, 
    # overriding whatever sub-optimal weights EarlyStopping might have left in RAM
    if os.path.exists(model_path):
        print(f"Loading absolutely best weights from {model_path} into RAM.")
        model.load_weights(model_path)
    
    return history

def unfreeze_and_finetune(model, train_data, val_data, layers_to_unfreeze=10, epochs=10, learning_rate=1e-5, csv_log_path=None, model_path=None, initial_epoch=15):
    """
    Unfreeze the top layers of the model for fine-tuning.
    """
    print(f"Unfreezing top {layers_to_unfreeze} layers for fine-tuning...")
    
    for layer in model.layers[-layers_to_unfreeze:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            
    # Recompile with a lower learning rate
    model = compile_model(model, learning_rate=learning_rate)
    
    # Train again with BackupAndRestore for Phase 2
    backup_name = model_path if model_path else model.name
    initial_acc = get_best_metric_from_csv(csv_log_path, 'val_accuracy', 'max')
    
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(backup_dir=f"backup_phase2_{backup_name}")
    ]
    
    if model_path:
        if initial_acc is not None:
            callbacks.append(ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1, initial_value_threshold=initial_acc))
        else:
            callbacks.append(ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1))
        
    if csv_log_path:
        callbacks.append(tf.keras.callbacks.CSVLogger(csv_log_path, append=True))
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + epochs,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False
    )
    
    # GUARANTEE: Load the best weights from disk back into RAM
    if model_path and os.path.exists(model_path):
        model.load_weights(model_path)
    
    return history
