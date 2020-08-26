import os, glob



#モデルの作成
def make_model(input_shape = (150,150, 3), class_number = 5, dropout_rate = 0.5, summary = False):
    print("start making model...")
    print("using keras CNN")
    from keras import models, layers, optimizers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(class_number, activation='softmax'))

    if summary: model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    return model

def learn_model(model, train_dir, validation_dir, target_size = (150,150), epochs = 50,save_name = "9-nine-.h5"):
    from keras.preprocessing.image import ImageDataGenerator
    #学習データのかさ増し
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=target_size, batch_size=32, class_mode="categorical")
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=target_size, batch_size=32, class_mode="categorical")

    #学習/保存
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=epochs, validation_data=validation_generator, validation_steps=50)
    print("saving model...")
    model.save(save_name)
    return history

def show_result(history):
    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def main(now_dir):
    learn_data_dir = os.path.join(now_dir, "learn_data")
    train_dir = os.path.join(learn_data_dir, 'train')
    validation_dir = os.path.join(learn_data_dir, 'validation')

    model = make_model()
    history = learn_model(model, train_dir, validation_dir, epochs=1)
    show_result(history)

if __name__ == "__main__":
    now_dir = os.getcwd()
    main(now_dir)