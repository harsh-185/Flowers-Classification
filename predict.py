from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

classes = ['Daffodil','Snowdrop', 'Lily Valley', 'Bluebell',
            'Crocus', 'Iris', 'Tigerlily', 'Tulip',
            'Fritillary', 'Sunflower', 'Daisy', 'Colts Foot',
             'Dandelalion', 'Cowslip', 'Buttercup', 'Windflower',
             'Pansy', 'Rose']
flowers={}

for i in range(18):
    flowers[i]=classes[i] 

    
model1 = load_model('Flowers/flower_classification_vgg19.h5')

image_path = 'Flowers/prediction'

testing_datagen = ImageDataGenerator(rescale = 1./255,validation_split=0)

img = testing_datagen.flow_from_directory(directory = image_path, target_size=(224, 224))
print(img)

prediction = (model1.predict(img))
print(prediction)

print(flowers[np.argmax(model1.predict(img))])
