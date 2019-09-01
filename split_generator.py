import os
import argparse
import numpy as np

if __name__ == '__main__':
	num_maps = int(input('Number of datapoints: '))

	map_ids = os.listdir('./json')
	map_ids = [map_id.split('.')[0] for map_id in map_ids]

	map_ids = np.random.choice(map_ids, num_maps, replace=False)

	save_path = "./" + str(num_maps) + ".txt"
	np.savetxt(save_path, map_ids, fmt="%s")
