import tensorflow as tf
from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(18, activation='softmax'))

optimizer = RMSprop(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_set = train_datagen.flow_from_directory(
    'training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_set = train_datagen.flow_from_directory(
    'training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'testing',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

history = model.fit(
    train_set,
    epochs=80,
    validation_data=val_set
)

loss, accuracy = model.evaluate(test_set)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

model.save('flower_classification_vgg19.h5')
