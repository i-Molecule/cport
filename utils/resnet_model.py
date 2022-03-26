from resnet_builder.resnet import ResnetBuilder

import tensorflow as tf

from numpy import rot90
import numpy as np

def get_model():
    L2 = 0.0001

    model = ResnetBuilder(name="ResNet18", mode="3D")
    input_layer = tf.keras.layers.Input((40, 40, 40, 7))

    print('resnet model was builded')
    big_model = tf.keras.models.Sequential(model.layers[:-1])
    big_model.add(tf.keras.layers.Flatten())
    big_model.add(tf.keras.layers.Dense(1000, activation='relu',\
        kernel_regularizer=tf.keras.regularizers.l2(L2)))
    big_model.add(tf.keras.layers.Dense(500, activation='relu',\
        kernel_regularizer=tf.keras.regularizers.l2(L2)))
    big_model.add(tf.keras.layers.Dense(200, activation='relu',\
        kernel_regularizer=tf.keras.regularizers.l2(L2)))
    big_model.add(tf.keras.layers.Dense(100, activation='relu',\
        kernel_regularizer=tf.keras.regularizers.l2(L2)))
    big_model.add(tf.keras.layers.Dense(1))

    big_model(input_layer)
    #return big_model
    custom_model = CustomModel(big_model.input, big_model.outputs)

    return custom_model

class CustomModel(tf.keras.Model):

    def predict_on_batch(self, x):
        if self.average_over_rotations:
            y_preds = []
            all_orientations = rotations24(x)
            for i, x_rotated in enumerate(all_orientations):
                prediction = self(x_rotated, training=False)
                y_preds.append(prediction)

            y_preds = np.mean(y_preds, axis=0)
            #y_preds = tf.convert_to_tensor(y_preds)
            
            return y_preds
        else:
            return super().predict_on_batch(x)

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        
        if self._average_over_rotations:
            y_preds = []
            all_orientations = rotations24(x)
            for i, x_rotated in enumerate(all_orientations):
                prediction = self(x_rotated, training=False)
                y_preds.append(prediction)
        
            y_preds = np.mean(y_preds, axis=0)
            y_preds = tf.convert_to_tensor(y_preds)
        else:
            y_preds = self(x, training=False)
        
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_preds, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_preds)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).

        return {m.name: m.result() for m in self.metrics}

    @property
    def average_over_rotations(self):
        return self._average_over_rotations
    
    @average_over_rotations.setter
    def average_over_rotations(self, flag):
        print("Averaging over 24 possible grid orientations: ", flag)
        self.average_over_rotations = flag


def rotations24(polycube):
    """List all 24 rotations of the given 3d array"""
    def rotations4(polycube, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        for i in range(4):
             yield rot90(polycube, i, axes)

    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    yield from rotations4(polycube, (2,3))

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotations4(rot90(polycube, 2, axes=(1,3)), (2,3))

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotations4(rot90(polycube, axes=(1, 3)), (1,2))
    yield from rotations4(rot90(polycube, -1, axes=(1,3)), (1,2))

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotations4(rot90(polycube, axes=(1,2)), (1,3))
    yield from rotations4(rot90(polycube, -1, axes=(1,2)), (1,3))

