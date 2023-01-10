import numpy as np
import pickle

with open('modifitest.txt', 'r') as f:
	data_list = f.read().split("\n")
	length = len(data_list)
	print(length)
	mm = length//15
	mmm = mm//30
	print(mmm)
	rere = np.resize(data_list, (mmm, 30, 15))
	print(rere.shape)
	
	
with open('modofi.p', 'wb') as ff:
	pickle.dump(rere,ff)

with open('modofi.p', 'rb') as ff:
	d2 = pickle.load(ff)

	data_text = open('modified.txt', 'w', encoding='utf-8', newline='\n')
	data_text.write(str(d2))
	data_text.close()
	
