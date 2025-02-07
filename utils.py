import random
import torch
import numpy as np
import os

def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)
	
# def get_random_computation_time() -> float:
# 	random_time = []
# 	# 0 rank=1
# 	random_time.append(np.random.uniform(0.05, 0.09)) # 100
# 	# 1 rank=2
# 	random_time.append(np.random.uniform(0.08, 0.12)) # 50
# 	# 2 rank=3
# 	random_time.append(np.random.uniform(0.2, 0.3)) # 200
# 	# 3 rank=4
# 	random_time.append(np.random.uniform(0.08, 0.12)) # 50
# 	# 4 rank=5
# 	random_time.append(np.random.uniform(0.02, 0.06)) # 100
# 	# 5 rank=6
# 	random_time.append(np.random.uniform(0.05, 0.09)) # 300
# 	# 6 rank=7
# 	random_time.append(np.random.uniform(0.02, 0.06)) # 200
# 	# 7 rank=8
# 	random_time.append(np.random.uniform(0.2, 0.3)) # 100
# 	# 8 rank=9
# 	random_time.append(np.random.uniform(0.05, 0.09)) # 100
# 	# 9 rank=10
# 	random_time.append(np.random.uniform(0.3, 0.5)) # 10
# 	return random_time

def get_random_computation_time() -> float:
	random_time = []
	'''
	#bert
	# 0 rank=1
	random_time.append(0.75) # 100
	# 1 rank=2
	random_time.append(1.5) # 50
	# 2 rank=3
	random_time.append(0.375) # 200
	# 3 rank=4
	random_time.append(1.5) # 50
	# 4 rank=5
	random_time.append(0.75) # 100
	# 5 rank=6
	random_time.append(0.25) # 300
	# 6 rank=7
	random_time.append(0.375) # 200
	# 7 rank=8
	random_time.append(0.75) # 100
	# 8 rank=9
	random_time.append(0.75) # 100
	# 9 rank=10
	random_time.append(7.5) # 10
	return random_time
	'''
	'''
	#cifar、mnist
 	# 0 rank=1
	random_time.append(0.09) # 100
	# 1 rank=2
	random_time.append(0.18) # 50
	# 2 rank=3
	random_time.append(0.045) # 200
	# 3 rank=4
	random_time.append(0.18) # 50
	# 4 rank=5
	random_time.append(0.09) # 100
	# 5 rank=6
	random_time.append(0.03) # 300
	# 6 rank=7
	random_time.append(0.045) # 200
	# 7 rank=8
	random_time.append(0.09) # 100
	# 8 rank=9
	random_time.append(0.09) # 100
	# 9 rank=10
	random_time.append(0.9) # 10
	return random_time
	'''
	#audiomnist
 	# 0 rank=1
	random_time.append(0.04) # 100
	# 1 rank=2
	random_time.append(0.08) # 50
	# 2 rank=3
	random_time.append(0.02) # 200
	# 3 rank=4
	random_time.append(0.08) # 50
	# 4 rank=5
	random_time.append(0.04) # 100
	# 5 rank=6
	random_time.append(0.013) # 300
	# 6 rank=7
	random_time.append(0.02) # 200
	# 7 rank=8
	random_time.append(0.04) # 100
	# 8 rank=9
	random_time.append(0.04) # 100
	# 9 rank=10
	random_time.append(0.4) # 10
	return random_time

if __name__ == "__main__":
	set_seed()
	for i in range(10):
		print(get_random_computation_time())
	for i in range(10):
		print(get_random_computation_time())