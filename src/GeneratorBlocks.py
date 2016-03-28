import numpy as np

class Blocks:
	def __init__(self):
		self.blocks = []
		self.currPos = -1

	def addBlock(self, block):
		self.blocks.append(block)
				
	def updateWork(self, e1, s2):
		 for b in self:
		 	rank = b.rank
		 	b.setWork(rank <= e1, rank >= s2)
		 	
	def hasRank(self, rank):
		for b in self:
			if b.rank == rank:
				return True
	
	def getBlock(self, rank):
		for b in self:
			if b.rank == rank:
				return b
		return None
		

	def __iter__(self):
		self.currPos=len(self.blocks)
		return self
	def next(self):
		if self.currPos <= 0:
			raise StopIteration
		else:
			self.currPos -= 1
			return self.blocks[self.currPos]