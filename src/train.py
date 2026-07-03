import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

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
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.BackupAndRestore(backup_dir=f"backup_phase1_{model_path}")
    ]
    
    if csv_log_path:
        callbacks.append(tf.keras.callbacks.CSVLogger(csv_log_path, append=True))
    
    print(f"Starting training for {model.name}...")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def unfreeze_and_finetune(model, train_data, val_data, layers_to_unfreeze=10, epochs=10, learning_rate=1e-5, csv_log_path=None):
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
    callbacks = [
        tf.keras.callbacks.BackupAndRestore(backup_dir=f"backup_phase2_{model.name}")
    ]
    
    if csv_log_path:
        callbacks.append(tf.keras.callbacks.CSVLogger(csv_log_path, append=True))
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history
