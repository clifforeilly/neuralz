
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras import backend as K
from datetime import datetime

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation="sigmoid"))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('C:\\Users\\cliff\\Downloads\\images\\training_set', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('C:\\Users\\cliff\\Downloads\\images\\test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator(training_set, steps_per_epoch=80, epochs=1, validation_data=test_set, validation_steps=2000)


test_image = image.load_img('C:\\Users\\cliff\\Downloads\\images\\c_t_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'creeper'
else:
    prediction = 'tree'

print(prediction)

export_path = 'C:\\Users\\cliff\\Downloads\\images\\model'

server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)

d = datetime.now().strftime("%Y%m%d%H%M%S")
classifier.save(export_path+"\\model_" + d + ".hd5")


builder = tf.saved_model.builder.SavedModelBuilder(export_path + "\\1")
signature = tf.saved_model.predict_signature_def(inputs={"images": classifier.input}, outputs={"scores": classifier.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={"predict": signature})

builder.save()

print(classifier.summary())

