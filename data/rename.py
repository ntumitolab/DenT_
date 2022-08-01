import os
import glob

path = 'data/AC16_2D/test/DNA'
files = os.listdir(path)

for index, file in enumerate(files):
	os.rename(os.path.join(path,file), os.path.join(path, ''.join([str(index), '.tif'])))
