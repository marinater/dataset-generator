import os
import json
import cv2
import argparse
import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from scipy.signal import convolve2d
import random
from tqdm import tqdm
import json

meter2pixel = 100
border_pad = 20

ALLOWED_ROOM_TYPES = {'Hallway', 'Gym', 'Office', 'Room', 'Kitchen', 'Hall', 'Dining_Room', 'Living_Room', 'Bedroom', 'Garage', 'Storage', 'Lobby', 'Toilet', 'Child_Room', 'Bathroom'}

ALLOWED_ROOM_TYPES = {a: i + 1 for i,a in enumerate(ALLOWED_ROOM_TYPES)}

def draw_map(file_name, json_path):
	with open(json_path + '/' + file_name + '.json') as json_file:
		json_data = json.load(json_file)

	verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
	x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
	cnt_map = np.zeros((y_max - y_min + border_pad * 2,
						x_max - x_min + border_pad * 2))

	verts[:, 0] = verts[:, 0] - x_min + border_pad
	verts[:, 1] = verts[:, 1] - y_min + border_pad

	cv2.drawContours(cnt_map, [verts], 0, 255, -1)
	cnt_map = cnt_map.astype(np.int)

	tp_map = np.full(cnt_map.shape, 1, dtype=np.uint8)

	fill_val = 2

	for category, rooms in json_data['room_category'].items():
		for room in rooms:
			bbox_tp = (np.array(room) * meter2pixel).astype(np.int)
			bbox = [np.max([bbox_tp[0] - x_min + border_pad, 0]),
					np.max([bbox_tp[1] - y_min + border_pad, 0]),
					np.min([bbox_tp[2] - x_min + border_pad, cnt_map.shape[1]]),
					np.min([bbox_tp[3] - y_min + border_pad, cnt_map.shape[0]])]

			tp_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = fill_val
			fill_val += 1

	tp_map[cnt_map == 0] = 255

	return tp_map * 20

def determineConnectivity(img, val_a, center_a, val_b, center_b):
	kernels = {
		'north' : [[val_a, val_a],[val_b,val_b]],
		'south' : [[val_b, val_b],[val_a,val_a]],
		'east' : [[val_b, val_a],[val_b,val_a]],
		'west' : [[val_a, val_b],[val_a,val_b]]
	}

	out = np.zeros(img.shape, dtype=np.uint8)
	out[img == val_a] = 100
	out[img == val_b] = 200

	k = np.array([[-1,-1,-1],[-1,12,-1],[-1,-1,-1]])
	out = convolve2d(out, k, 'same')
	
	door = np.zeros(img.shape, dtype=np.uint8)
	door[out==100] = 255

	if len(np.unique(door)) == 1:
		return 0
	else: return 1

def generateAdjacency(img):
	out = np.copy(img)

	nodeLabels, nodeSizes = np.unique(img, return_counts=True)
	nodeLabels = nodeLabels[:-1]
	nodeSizes = nodeSizes[:-1]

	N = len(nodeLabels)

	centroids = center_of_mass(img, img, nodeLabels)
	centroids = [ (int(b), int(a)) for a,b in centroids ]

	NODE_INFO = { str(i) : {'connections':[], 'square_footage':nodeSizes[i].item()} for i in range(len(nodeLabels))}


	for index, centroid in enumerate(centroids):
		cv2.circle(out, centroid, 4, 255, 4)
		cv2.putText(out, str(nodeSizes[index]), (centroids[index][0] + 5,centroids[index][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

	for index_a, val_a in enumerate(nodeLabels):
		for index_b, val_b in enumerate(nodeLabels[index_a + 1:]):
			index_b += 1 + index_a

			connectivity = determineConnectivity(img, val_a, centroids[index_a], val_b, centroids[index_b])

			if connectivity == 1:
				NODE_INFO[str(index_a)]['connections'].append(index_b)
				NODE_INFO[str(index_b)]['connections'].append(index_a)
				cv2.line(out, centroids[index_a], centroids[index_b], 255, 4)

	return out, NODE_INFO

if __name__ == '__main__':
	in_map = input('Map ID file (ENTER for default): ').strip()
	if in_map == '': in_map = '100.txt'
	elif in_map.isdigit(): in_map += '.txt'

	json_path = os.path.abspath(os.path.join(os.getcwd(), './json'))
	map_file = os.path.abspath(os.path.join(os.getcwd(), in_map))
	save_path = os.path.abspath(os.path.join(os.getcwd(), './out'))

	map_ids = np.loadtxt(map_file, str)

	if not os.path.exists(save_path):
		os.mkdir(save_path)

	for map_id in tqdm(map_ids):
		groundTruth = draw_map(map_id, json_path)
		img, data = generateAdjacency(groundTruth)

		with open(save_path + "/" + map_id + '.json', 'w') as f:
			json.dump(data, f)
		
		cv2.imwrite(save_path + "/" + map_id + '.png', img)