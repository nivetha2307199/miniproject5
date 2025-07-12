import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
import seaborn as sns # type: ignore
#Data Preprocessing
train_dir = r'C:\pro2\.venv\Scripts\Image_dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train'
val_dir = r'C:\pro2\.venv\Scripts\Image_dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val'
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, 
                                              target_size=img_size, 
                                              class_mode='categorical', 
                                              batch_size=batch_size)
val_gen = val_datagen.flow_from_directory(val_dir, 
                                          target_size=img_size,
                                            class_mode='categorical', 
                                            batch_size=batch_size)
#print(train_gen)

# Load base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('best_fish_model.h5', save_best_only=True),
    ReduceLROnPlateau(patience=2)
]
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)
#Evaluation and visualization
# Accuracy/Loss Plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()


# Confusion Matrix
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_gen.classes

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


print(classification_report(y_true, y_pred_classes, target_names=list(val_gen.class_indices.keys())))
model.save('fish_classifier_vgg16.h5')