"""
This module provides APIs for logging loss from other parts of the program.
It also contains the implementation of the loss tracker itself.
The aim is to keep the modifications to other parts minimal.
"""
import tensorflow as tf
from tensorflow import keras
import modelUtils
from layer_namer import _layer_name
import numpy as np
import preprocessor
import matplotlib.pyplot as plt
import os

import logging
logger = logging.getLogger('lossTracker')

lossTracker = None
trackMode = "STEP"
SAVE_DIR = "trackerPlot"

# Currently supported: "AVG" or "SUM"
# This defines what happend when plotting loss against observable values
MODE = "AVG"

EVENT_ELEMENT_LABELS = []

# Unfortunately the process of evaluating the model against each individual event is really slow.
# It is however necesssary since we need to infer each individual loss here.
# Thus the compromise is to sample a set amount of events.
# The events are already randomly arranged, so taking the first N events should suffice.
TRACKING_SAMPLING_N = 10 # currently set to 100 for debugging purposes.

class LossTracker():
	def __init__(self, session_name)->None:
		self.session_name = session_name
		self.data = []
		self.loss = []
		self.weight = []
		self.iteration = 0
		self.run = 0

		# safe to call getObservables here with the assumption that lossTracker is intialized
		# in modelUtils.train_model, which is long after preprocessing has finished.
		self.order = preprocessor.get().getObservables()
	def newIteration(self, iteration)->None:
		self.iteration = iteration
	def newRun(self, run)->None:
		self.run = run
	def updateSession(self, session_name)->None:
		pass
	def evaluateLoss(self, model, data):
		"""
		This function is designed to be invoked by the model during train step.
		It will forward itself and the current batch of data here for evaluating the loss values.
		"""
	def get():
		return [], []
	def setOrder(self, order):
		"""
		Strictly speaking, this functionality is not entirely necessary. It is for labelling what each
		element of the event means physically.
		"""
		self.order = order
	def getObservableLoss(self, elementName):
		"""
		Returns the 1d list of observable value and the corresponding 1d list of loss.
		"""
		idx = 0
		for i, ob in enumerate(self.order):
			if ob == elementName:
				idx = i
		
		return self.data[:, idx], self.loss
	def plotLoss(self):
		"""
		Plotting the currently recorded loss to file. Implementation can differ depending
		on what is really being tracked and how sessions are divided.
		"""
		pass

"""
Instead of puting "if enable" checks everywhere in code, the loss tracker can be instead
disabled by creating a dummy instance that does not do any thing.
"""
class DisabledTracker(LossTracker):
	pass

# class InterEpochLossTracker(LossTracker):
# 	def evaluateLoss(self, model, data):
# 		# unpack data
# 		inputs, outputs, weights = data[0], data[1], data[2]

# 		# generate key names for simple access
# 		input_keys = [_layer_name(i, "input") for i in range(modelUtils.n_models_in_parallel)]
# 		output_keys = [_layer_name(i, "output") for i in range(modelUtils.n_models_in_parallel)]

# 		input_frame, output_frame, weight = {}, {}, 0

# 		for i in range((inputs[input_keys[0]].shape)): # how many events we have
# 			for n in modelUtils.n_models_in_parallel:
# 				input_frame[_layer_name(n, "input")] = inputs[_layer_name(n, "input")][i]
# 				output_frame[_layer_name(n, "output")] = outputs[_layer_name(n, "output")][i]
# 				weight = weights[i]

class StepLossTracker(LossTracker):
	def appendLoss(self, data, loss, weight):
		if len(self.loss) != 0 and len(self.data) != 0:
			self.loss = np.concatenate((self.loss, loss))
			self.data = np.concatenate((self.data, data))
			self.weight = np.concatenate((self.weight, weight))
		else:
			self.loss = loss
			self.data = data
			self.weight = weight

	def updateSession(self, session_name) -> None:
		self.session_name = session_name
		self.loss = []
		self.data = []
		self.weight = []

	def evaluateLoss(self, model, data):
		logger.info("Beginning loss evaluation")
		inputs, outputs, weights = data[0], data[1], np.array(data[2])
		
		# generate key names for simple access
		event_count = (np.shape(weights))[1]

		loss = np.zeros((modelUtils.n_models_in_parallel, event_count))

		input_frame, output_frame, weight_frame = {}, {}, []
		for i in range(TRACKING_SAMPLING_N): # how many events we have
			for n in range(modelUtils.n_models_in_parallel):
				column = inputs[_layer_name(n, "input")][i]
				input_frame[_layer_name(n, "input")] = np.reshape(column, (1,) + np.shape(column))
				column = outputs[_layer_name(n, "output")][i]
				output_frame[_layer_name(n, "output")] = np.reshape(column, (1,) + np.shape(column))

			# case 1 model: model.evaluate returns single loss scalar
			# case multilpe models: model.evaluate returns array of [total loss, model 1 loss, ..., model n loss, model 1 accuracy, ..., model n accuracy]
			# check model.metrics_names for details
			if modelUtils.n_models_in_parallel == 1:
				loss[:, i] = (model.evaluate(x = input_frame, y = output_frame, verbose = 0))[0]
			else:
				loss[:, i] = (model.evaluate(x = input_frame, y = output_frame, verbose = 0))[1 : 1 + modelUtils.n_models_in_parallel]
			if (i % (TRACKING_SAMPLING_N / 100) == 0):
				msg = str(i / (TRACKING_SAMPLING_N / 100)) + "% done\n"
				logger.info(msg)
		self.appendLoss(data[0][_layer_name(0, "input")], loss, weights)

	def get(self):
		return self.data, self.loss, self.weight
	
	def plotLoss(self):
		for ob_name in self.order:
			data, loss = self.getObservableLoss(ob_name)
			loss = np.average(loss, weights = self.weight, axis=0) # Taking the weighted average of parallel models
			plt.clf() # Clear any previously plotted graph

			fig, axs = plt.subplots(3)

			# counting how many events are in each bin
			event_cnt, bin_edges, patches = axs[2].hist(data)
			axs[2].set_title(ob_name + " event count")
			axs[2].set_xlabel(ob_name)
			axs[2].set_ylabel("number of events")

			# loss against observable value
			n, bin_edges, patches =  axs[0].hist(data, weights = loss)
			axs[0].set_title(ob_name + " loss distribution")
			axs[0].set_xlabel(ob_name)
			axs[0].set_ylabel("loss")

			# saving figure
			# TODO: Move this into output dir in the future
			# plt.savefig(os.path.join("trackerPlot", ob_name+"_"+self.session_name+"_loss.png"))

			# plt.clf()

			# plotting average loss 

			axs[1].bar((bin_edges[0:-1] + bin_edges[1:]) / 2, n / event_cnt)
			axs[1].set_title(ob_name + "average loss")
			axs[1].set_xlabel(ob_name)
			axs[1].set_ylabel("loss")

			# saving figure
			# TODO: Move this into output dir in the future

			try_create = lambda path: os.makedirs(path) if not os.path.isdir(path) else None

			save_path = SAVE_DIR
			try_create(save_path)
			current_run_name = "run_" + str(self.run)
			save_path = os.path.join(save_path, current_run_name)
			try_create(save_path)
			current_iter_name = "iteration_" + str(self.iteration)
			save_path = os.path.join(save_path, current_iter_name)
			try_create(save_path)
			plt.savefig(os.path.join(save_path, ob_name+"_"+self.session_name+".png"))
		
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
		lossTracker = StepLossTracker(session_name) if trackingStep() else InterEpochLossTracker(session_name);
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
	if lossTracker == None:
		if trackingStep():
			lossTracker = StepLossTracker("Default Session Name")
		# elif interEpochTracking():
		# 	lossTracker = InterEpochLossTracker("Default Session Name")
		elif trackingDisabled():
			lossTracker = DisabledTracker("Default Session Name")

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

def trackingDisabled()->bool:
	return trackMode == "DISABLE"