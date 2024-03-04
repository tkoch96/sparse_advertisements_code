import numpy as np
class Foo:
	def __init__(self):
		pass
	def f(self):
		self.x=np.array([4])
		# self.y=getattr(self,'x')
		self.y = self.x
		self.x[0] += 1
		print("{} {}".format(self.x,self.y))

f = Foo()
f.f()