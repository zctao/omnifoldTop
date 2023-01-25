import tensorflow as tf
from tensorflow import keras

class LossTrackerModel(keras.models.Model):
	def train_step(self, data):
		return super().train_step(data)