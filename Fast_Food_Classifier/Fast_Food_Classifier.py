from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception, DenseNet121, MobileNetV2, MobileNetV3Small, MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, Accuracy,F1Score
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def data_loading(data_dir, target_size=(256,256), batch_size=128, validation_split=0.25, rotation_range = 20, width_shift_range = 0.2, height_shift_range = 0.2):
  datagen=ImageDataGenerator(rescale=1./255,
                             rotation_range=rotation_range,
                             height_shift_range=height_shift_range,
                             width_shift_range=width_shift_range,
                             zoom_range=0.2,
                             horizontal_flip=0.15,
                             validation_split=validation_split)


  train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='training'
    )

  val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation'
    )

  return train_generator, val_generator


def model(input_shape, num_classes):

  base_model=MobileNet(weights='imagenet',include_top=False)
  x=GlobalAveragePooling2D()(base_model.output)
  x=Dense(1024,activation='relu')(x)
  x=Dropout(0.2)(x)
  x=Dense(1024,activation='relu')(x) #dense layer 2
  x=Dense(512,activation='relu')(x) #dense layer 3
  x=Dropout(0.5)(x)
  preds=Dense(6,activation='softmax')(x) #final layer
  model = Model(inputs=base_model.input, outputs=preds)

  model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", Precision(), Recall()]
  )

  return model


def training(model, train_generator, val_generator, log_folder):
    checkpoint = ModelCheckpoint(
        os.path.join(log_folder, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1
    )
    tensorboard_callback = TensorBoard(
        log_dir=log_folder,
        histogram_freq=1
    )

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )

    return history


def evaluation(model, val_generator):
    test_loss, test_acc = model.evaluate(val_generator)
    print("Test accuracy: {:.2%}".format(test_acc))

    test_preds = model.predict(val_generator)
    test_labels = np.argmax(test_preds, axis=-1)

    conf_mat = confusion_matrix(val_generator.classes, test_labels)
    print("Confusion matrix:")
    print(conf_mat)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes), rotation=45)
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Plot the training and validation accuracy and loss
    acc_plt = (plt.figure(figsize=(8, 6)),
               plt.subplot(1, 2, 1),
               plt.plot(history.history['accuracy'], label='Training Accuracy'),
               plt.plot(history.history['val_accuracy'], label='Validation Accuracy'),
               plt.legend(),
               plt.title('Accuracy'))

    los_plt = (plt.subplot(1, 2, 2),
               plt.plot(history.history['loss'], label='Training Loss'),
               plt.plot(history.history['val_loss'], label='Validation Loss'),
               plt.legend(),
               plt.title('Loss'),
               plt.show())
if __name__ == "__main__":
    # Load the dataset
    data_dir = "/content/drive/MyDrive/Colab_Notebooks/fast"

    # Define the number of classes
    num_classes = 6

    # Load and split the data
    train_generator, val_generator = data_loading(data_dir)

    # Build the model
    input_shape = (256, 256, 3)
    model = model(input_shape, num_classes)

    # Define the log folder for callbacks
    log_folder = 'logs'

    # Train the model
    history = training(model, train_generator, val_generator, log_folder)

    # Evaluate the model
    evaluation(model, val_generator)

    # Save the trained model
    model.save("/content/drive/MyDrive/trained_model.h5")

