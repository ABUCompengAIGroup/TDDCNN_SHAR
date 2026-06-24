
# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense, Dropout, BatchNormalization, Bidirectional, LSTM, Add
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, num_classes):

    inputs = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same', dilation_rate=2))(inputs)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(BatchNormalization())(x)

    residual = x
    residual = TimeDistributed(MaxPooling2D((2,2)))(residual)
    residual = TimeDistributed(Conv2D(64, (1,1), padding='same'))(residual)

    x = TimeDistributed(Conv2D(64, (3,3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)

    x = Add()([x, residual])

    x = TimeDistributed(Flatten())(x)

    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
