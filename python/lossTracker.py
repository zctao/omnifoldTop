"""
This module provides APIs for logging loss from other parts of the program.
It also contains the implementation of the loss tracker itself.
The aim is to keep the modifications to other parts minimal.
"""
import tensorflow as tf
from tensorflow import keras
import modelUtils
from layer_namer import _layer_name

lossTracker = None
trackMode = "STEP"

class LossTracker():
	def __init__(self, session_name)->None:
		self.session_name = session_name
		pass
	def updateSession(self, session_name)->None:
		pass
	def evaluateLoss(self, model, data):
		"""
		This function is designed to be invoked by the model during train step.
		It will forward itself and the current batch of data here for evaluating the loss values.
		"""

class InterEpochLossTracker(LossTracker):
	def evaluateLoss(self, model, data):
		# unpack data
		inputs, outputs, weights = data[0], data[1], data[2]

		# generate key names for simple access
		input_keys = [_layer_name(i, "input") for i in range(modelUtils.n_models_in_parallel)]
		output_keys = [_layer_name(i, "output") for i in range(modelUtils.n_models_in_parallel)]

		input_frame, output_frame, weight = {}, {}, 0
		print(inputs[input_keys[0]].shape)

		for i in range((inputs[input_keys[0]].shape)): # how many events we have
			for n in modelUtils.n_models_in_parallel:
				input_frame[_layer_name(n, "input")] = inputs[_layer_name(n, "input")][i]
				output_frame[_layer_name(n, "output")] = outputs[_layer_name(n, "output")][i]
				weight = weights[i]
			print(i)
			print(input_frame)
			print(output_frame)
			print(weight)

class StepLossTracker(LossTracker):
	def evaluateLoss(self, model, data):
		return super().evaluateLoss(model, data)


def getTrackerInstance(session_name, refresh)->LossTracker:
	"""
	Arguments
	---------
	session_name: str
		name of the current tracking session. Loss from different iterations across different
		runs shouldn't be tracked together. We are interested in loss generated from a relatively
		homogenous training step.
	refresh: boolean
		if refresh is True, a new tracker instance from given session_name is created and returned. Otherwise, session_name
		is passed to the tracker instance and handled there.
	
	Returns
	-------
	tracker: LossTracker
		whether the returned tracker instance is a new instance depends on refresh flag
	"""
	if refresh:
		lossTracker = LossTracker(session_name)
	else:
		lossTracker.updateSession(session_name)
	return lossTracker

def getTrackerInstance()->LossTracker:
	"""
	Returns
	-------
	tracker: LossTracker
		the current tracker instance. A new default tracker will be initiated if None.
	"""
	global lossTracker
	if lossTracker == None: lossTracker = LossTracker("Default Session Name");
	return lossTracker

def getTrackMode()->str:
	return trackMode

def trackingStep()->bool:
	if trackMode == "STEP":
		return True
	else:
		return False
	
def interEpochTracking()->bool:
	if trackMode == "EPOCH":
		return True
	else:
		return False