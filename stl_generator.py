import os
import json
import numpy as np
import json
from stl import mesh
from tqdm import tqdm

meter2pixel = 100
border_pad = 20
HEIGHT = 50

def vert2stl(raw_verts, filename):
	N = len(raw_verts)
	
	vertices = np.array([ [x, 0, z] for x,z in raw_verts] + [ [x, HEIGHT, z] for x,z in raw_verts])
	faces = np.array([ [i , (i + 1)%N, N + i] for i in range(N) ] + [ [(N + i), N + ((i + 1)%N), (i + 1) %N] for i in range(N) ])

	cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(faces):
		for j in range(3):
			cube.vectors[i][j] = vertices[f[j],:]

	cube.save('./out/' + map_id + '.stl')

def getVerts(file_name, json_path):
	with open(json_path + '/' + file_name + '.json') as json_file:
		json_data = json.load(json_file)

	verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
	x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
	cnt_map = np.zeros((y_max - y_min + border_pad * 2,
						x_max - x_min + border_pad * 2))

	verts[:, 0] = verts[:, 0] - x_min + border_pad
	verts[:, 1] = verts[:, 1] - y_min + border_pad

	return verts

if __name__ == '__main__':
	in_map = input('Map ID file (ENTER for default): ').strip()
	if in_map == '': in_map = '100.txt'
	elif in_map.isdigit(): in_map += '.txt'

	json_path = os.path.abspath(os.path.join(os.getcwd(), './json'))
	map_file = os.path.abspath(os.path.join(os.getcwd(), in_map))
	save_path = './out/'

	map_ids = np.loadtxt(map_file, str)

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	for map_id in tqdm(map_ids):
		vert2stl(getVerts(map_id, json_path), save_path + map_id + '.stl')