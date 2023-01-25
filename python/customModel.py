import tensorflow as tf
from tensorflow import keras

class LossTrackerModel(keras.models.Model):
	def train_step(self, data):
		"""
		Overriding the training step from parent keras.models.Model gives a window during each traning step to take a look
		at the input data and the output loss, which can be passed to the tracker model for analysis.
		"""
		return super().train_step(data)