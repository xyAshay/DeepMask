from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


LEARNING_RATE = 1e-5
EPOCHS = 15
BATCH_SIZE = 32
OPTR = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)

x = np.load('X_RAW.npy')
Y = np.load('Y_RAW.npy')

print(x.shape)
print(Y.shape)

x_train, x_test, Y_train, Y_test = train_test_split(x, Y,
                                                    test_size=0.2, stratify=Y,
                                                    random_state=9
                                                   )
print('Training Set [x, Y]', x_train.shape, Y_train.shape)
print('Dev Set [x, Y]', x_test.shape, Y_test.shape)

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(100, 100, 3))
                       )

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(3,3), padding='valid')(headModel)
headModel = Flatten(name='Flatten')(headModel)
headModel = Dense(32, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False
model.compile(loss='binary_crossentropy', optimizer=OPTR, metrics=['accuracy'])

DeepFace = model.fit(x_train, Y_train, batch_size=BATCH_SIZE,
                     steps_per_epoch = len(x_train) // BATCH_SIZE,
                     validation_data=(x_test, Y_test),
                     validation_steps = len(x_test) // BATCH_SIZE,
                     epochs=EPOCHS
                    )
                    
model.save('DeepFaceMask.model', save_format="h5")

"""
plt.plot(DeepFace.history['loss'],'r',label='Training Loss')
plt.plot(DeepFace.history['val_loss'],label='Validation Loss')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.legend()
plt.show()

plt.plot(DeepFace.history['accuracy'],'r',label='Training Accuracy')
plt.plot(DeepFace.history['val_accuracy'],label='Validation Accuracy')
plt.xlabel('EPOCHS')
plt.ylabel('LOSS')
plt.legend()
plt.show()
"""
