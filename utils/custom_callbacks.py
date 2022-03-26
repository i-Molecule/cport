import tensorflow as tf
import gc

class GarbageCollector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        gc.collect()
        tf.keras.backend.clear_session()

    def on_batch_end(self, batch, logs={}):
        gc.collect()
        tf.keras.backend.clear_session()

def custom_mse(thr=0.0):
    thr=thr
    def mse(y_true, y_pred):
        #thr = THRESHOLD
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        difference = tf.abs(y_pred - y_true)
        
        sparse_difference = tf.where(difference > thr, difference, tf.zeros_like(difference))

        return tf.reduce_mean(tf.square(sparse_difference))
    
    return mse