import numpy as np
class Block:
	def __init__(self, A1, rank):
		self.A1 = A1
		self.A2 = 1j*A1
		self.work1 = False
		self.work2 = False
		self.rank = rank
		
	def setWork1(self, work1 = True):
		self.work1 = True
	def setWork2(self, work2 = True):
		self.work2 = True
		
	def setWork(self, work1, work2):
		self.work1 = work1
		self.work2 = work2
	
	def getWork(self):
		return work1, work2
		
	def getWork1(self):
		return work1
		
	def getWork2(self):
		return work2