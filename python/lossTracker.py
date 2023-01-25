"""
This module provides APIs for logging loss from other parts of the program.
It also contains the implementation of the loss tracker itself.
The aim is to keep the modifications to other parts minimal.
"""

lossTracker = None

def getTrackerInstance(session_name, refresh):
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

class LossTracker():
	