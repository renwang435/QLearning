from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

class CNN():

    # IDEA: We crop the Space Invaders input image of size (210, 160) to (96, 96),
    # which should contain all the salient information. We also grayscale the image
    # (i.e. turn 3 channels into 1) as color should not affect the model. Finally,
    # we also stack a series of input frames together (tunable hyperparameter) to
    # more efficient parse semantic information from a sequence of game states.
    # We output a set of actions (likely to be one of moving left, moving right,
    # shooting, and not moving)

    def __init__(self, n_input_frames, n_output_actions):
        self.model = Sequential()
        self.model.add(Conv2D(16, (8, 8),
                              strides=(4, 4),
                              padding='valid',
                              activation='relu',
                              input_shape=(96, 96, n_input_frames)))
        self.model.add(Conv2D(32, (4, 4),
                              strides=(2, 2),
                              padding='valid',
                              activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256,
                             activation='relu'))
        self.model.add(Dense(n_output_actions))
        self.model.compile(loss='mse',
                           optimizer=RMSprop(lr=0.00025),
                           metrics=['accuracy'])
