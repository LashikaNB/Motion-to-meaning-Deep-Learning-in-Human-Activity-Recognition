# STEP 1: Set paths (for Kaggle)
dataset_path = '/kaggle/input/human-activity-recognition/'
df_csv_path = f'{dataset_path}/Training_set.csv'
df_folder = f'{dataset_path}/train'

# STEP 2: Load CSV
df = pd.read_csv(df_csv_path)
df['filename'] = '/kaggle/input/human-activity-recognition/train-20250414T123037Z-001/train/' + df['filename']

# STEP 3: Split Dataset
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# STEP 4: Data Generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# STEP 5: Build Model with EfficientNetB3
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

base_model = EfficientNetB3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# STEP 6: Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
checkpoint = ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss")

# STEP 7: Train Initial Model
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# STEP 8: Fine-Tune
for layer in base_model.layers[-100:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# STEP 9: Plot Training Curves
import matplotlib.pyplot as plt

history_dict = history_finetune.history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# STEP 10: Evaluation
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

preds = model.predict(test_gen)
y_pred = preds.argmax(axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_gen.class_indices,
            yticklabels=test_gen.class_indices)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

# STEP 11: Save Model & Class Names
model.save("final_model.keras")

import json
class_names = list(train_gen.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)
