import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# Checkpoint to save the best model
checkpoint = ModelCheckpoint('models/artgan.keras', 
                             save_best_only=True, 
                             monitor='val_loss', 
                             mode='min')

# Set image size and input shape
image_size = (150, 150)  # Match to your dataset image size
input_shape = (150, 150, 3)

# Data augmentation and preparation for training/validation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    validation_split=0.2  # Use a portion for validation
)

train_generator = train_datagen.flow_from_directory(
    'wikiart/',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'wikiart/',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Building the CNN Model
model = Sequential()

# Input Layer
model.add(InputLayer(input_shape=input_shape))

# First Convolutional Layer with L2 Regularization and Pooling
model.add(Conv2D(32, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer with L2 Regularization
model.add(Conv2D(64, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer with L2 Regularization
model.add(Conv2D(128, (3, 3), activation='relu',
                 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the 2D output to 1D for dense layers
model.add(Flatten())

# Fully Connected Dense Layer with Dropout
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting

# Output Layer for 27 classes (based on the problem statement)
model.add(Dense(27, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model Summary
model.summary()

# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5,
    callbacks=[checkpoint]  # Save the best model during training
)

# Evaluating the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Plotting training history (optional)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
