import numpy as np
import pickle

with open('C:/Users/yusuk/Desktop/Y/waseda/研究室/留学/tasks/classes/collective/collective-main/collective-main/src_skeleton_extraction/data/223_01_10_16_08_1673363335', 'r') as f:
	data_list = f.read().split()
print(len(data_list))
#data_list = data_list.remove('[')
#data_list.remove('[')
#data_list = data_list.remove(']')
#data_list.remove(']')
print(len(data_list))

count = 0
list_example2 = []
while count < len(data_list):
	if data_list[count] != '[' and data_list[count] != ']':
		list_example2.append(data_list[count])
	count += 1

data_list = list_example2


for i in range(len(data_list)):
	data_list[i] = data_list[i].replace('[', '')
	data_list[i] = data_list[i].replace(']', '')
	#print(type(data_list[i]))
	#print(data_list[i])
	data_list[i] = float(data_list[i])
	#print(type(data_list[i]))
	print(data_list[i])
	#print(i)

print(len(data_list))


length = len(data_list)
print(length)
mm = length//15
print(mm)
mmm = mm//30
print(mmm)

rere = np.resize(data_list, (mmm, 30, 15, 2))
print(rere)
print(rere.shape)
	
	
with open('modofi.p', 'wb') as ff:
	pickle.dump(rere,ff)

with open('modofi.p', 'rb') as ff:
	d2 = pickle.load(ff)

	data_text = open('C:/Users/yusuk/Desktop/Y/waseda/研究室/留学/tasks/classes/collective/collective-main/collective-main/src_skeleton_extraction/data/modified.txt', 'w', encoding='utf-8', newline='\n')
	data_text.write(str(d2))
	data_text.close()
	
