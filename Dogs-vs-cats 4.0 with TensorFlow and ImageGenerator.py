from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization

'''_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # директория с картинками котов для обучения
train_dogs_dir = os.path.join(train_dir, 'dogs')  # директория с картинками собак для обучения
validation_cats_dir = os.path.join(validation_dir, 'cats')  # директория с картинками котов для проверки
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # директория с картинками собак для проверки
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)'''
batch_size = 15
epochs = 20
IMG_HEIGHT = 128
IMG_WIDTH = 128
train_image_generator = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                ) # Генератор для тренировочных данных
validation_image_generator = ImageDataGenerator(rescale=1./255) # Генератор для проверочных данных
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory='train/',
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory='validation/',
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')
sample_training_images, _ = next(train_data_gen)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(sample_training_images[:5])
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.00001)
callbacks = [earlystop,learning_rate_reduction]
model.summary()
history = model.fit_generator(
    train_data_gen, 
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=5000//batch_size,
    steps_per_epoch=20000//batch_size,
    callbacks=callbacks
)
test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1)

test_generator = test_datagen.flow_from_directory(
        directory = 'test/',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

predict = model.predict_generator(test_generator,steps = 12500)
print(predict)
with open('test_result4.0.csv',  newline="", mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',',skipinitialspace=True, quoting=csv.QUOTE_NONE)
    test_writer.writerow(['id','label'])
    for j, elem in enumerate(predict):
        test_writer.writerow([j+1, np.argmax(elem)])
       
print('Success!!!')