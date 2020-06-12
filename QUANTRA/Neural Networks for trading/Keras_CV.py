from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense, Activation

def create_new_model(neurons=20,act_1='sigmoid',dropout_ratio=0.15):

    model=Sequential()
    model.add(Dense(neurons,
                    kernel_initializer='he_normal'
                   ,input_shape=(5,)
                   , bias_initializer='zeros'))
    model.add(Activation(act_1))
    model.add(Dropout(dropout_ratio))

    model.add(Dense(neurons*2, use_bias=True, kernel_initializer='he_normal'
                   , bias_initializer='zeros'))
    model.add(Activation(act_1))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(neurons*3, use_bias=True, kernel_initializer='he_normal'
                  , bias_initializer='zeros'))
    model.add(Activation(act_1))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(neurons*4, use_bias=True, kernel_initializer='he_normal'
                 , bias_initializer='zeros'))
    model.add(Activation(act_1))
    model.add(Dropout(dropout_ratio))
    model.add(Dense(neurons*5, use_bias=True, kernel_initializer='he_normal'
                 , bias_initializer='zeros'))
    model.add(Activation(act_1))
    model.add(Dropout(dropout_ratio))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


    return model