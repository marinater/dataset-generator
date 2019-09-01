import os
import json
import cv2
import argparse
import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from scipy.signal import convolve2d
from h2rgb import h2rgb
from tqdm import tqdm

meter2pixel = 30	
border_pad = 20

ALLOWED_ROOM_TYPES = {'Hallway', 'Gym', 'Office', 'Room', 'Kitchen', 'Hall', 'Dining_Room', 'Living_Room', 'Bedroom', 'Garage', 'Storage', 'Lobby', 'Toilet', 'Child_Room', 'Bathroom'}

ALLOWED_ROOM_TYPES = {a: i + 1 for i,a in enumerate(ALLOWED_ROOM_TYPES)}

def draw_map(file_name, json_path):
	with open(json_path + '/' + file_name + '.json') as json_file:
		json_data = json.load(json_file)

	verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
	x_max, x_min, y_max, y_min = np.max(verts[:, 0]), np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 1])
	footprint = np.full((y_max - y_min + border_pad * 2,
						x_max - x_min + border_pad * 2), 255, dtype=np.uint8)
	walls = np.zeros(footprint.shape, dtype=np.uint8)

	verts[:, 0] = verts[:, 0] - x_min + border_pad
	verts[:, 1] = verts[:, 1] - y_min + border_pad

	cv2.drawContours(footprint, [verts], 0, 0, -1)
	cv2.drawContours(walls, [verts], 0, 255, 5)

	rooms_map = np.full(footprint.shape, 0, dtype=np.uint8)
	fill_val = 1

	centers = []

	for category, rooms in json_data['room_category'].items():
		for room in rooms:
			bbox_tp = (np.array(room) * meter2pixel).astype(np.int)
			bbox = [np.max([bbox_tp[0] - x_min + border_pad, 0]),
					np.max([bbox_tp[1] - y_min + border_pad, 0]),
					np.min([bbox_tp[2] - x_min + border_pad, footprint.shape[1]]),
					np.min([bbox_tp[3] - y_min + border_pad, footprint.shape[0]])]

			centroid = ((bbox[1] + bbox[3]) // 2 , (bbox[0] + bbox[2]) // 2)
			centers.append( (category, centroid, fill_val))

			rooms_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = fill_val
			fill_val += 1

	footprint[footprint == 0] |= rooms_map[footprint == 0]
	footprint[walls == 255] = 255

	unknown = np.zeros(footprint.shape, dtype=np.uint8)
	unknown[footprint == 0] = 255

	ret, markers = cv2.connectedComponents(unknown)
	labels = np.unique(markers).nonzero()
	centroids = center_of_mass(markers, markers, labels)

	centroids = [ ( int(centroid[0][0]), int(centroid[0][1]) ) for centroid in centroids ]

	out = np.zeros(footprint.shape, dtype=np.uint8)

	unknown_centers = []
	for centroid in centroids:
		unknown_centers.append(('Unknown', centroid, fill_val))
		cv2.floodFill(out, np.pad(255 - unknown, 1, mode='constant'), centroid[::-1], fill_val)
		fill_val += 1

	for category, centroid, fill_val in centers:
		mask = np.full(footprint.shape, 255, dtype=np.uint8)
		mask[footprint == fill_val] = 0
		cv2.floodFill(out, np.pad(mask, 1, mode='constant'), centroid[::-1], fill_val)

	return out, centers + unknown_centers

def determineConnectivity(img, val_a, center_a, val_b, center_b):
	out = np.zeros(img.shape, dtype=np.uint8)
	out[img == val_a] = 100
	out[img == val_b] = 200

	labels, count = label(out)
	# k = np.array([[-1,-1,-1],[-1,12,-1],[-1,-1,-1]])
	# out = convolve2d(out, k, 'same')
	
	# door = np.zeros(img.shape, dtype=np.uint8)
	# door[out==100] = 255

	# cv2.waitKey()

	cp = np.copy(img)
	cv2.circle(cp, center_a, 4, 255, 4)
	cv2.circle(cp, center_b, 4, 255, 4)

	# cv2.imshow('orig', cp * 100)
	# cv2.imshow('door', labels.astype(np.uint8) * 100)
	# cv2.imshow('mask', out)

	cv2.waitKey()

	return count == 1

def generateAdjacency(img, stats):
	display = h2rgb(img)

	NODE_INFO = [
		{'connections':[], 'squareFootage': np.count_nonzero(img==fill_val), 'category': cat}
		for cat,_,fill_val in stats
	]

	for _, center,_ in stats:
		cv2.circle(display, center[::-1], 4, [255,255,255], 4)

	for index_a in range(len(stats) - 1):
		for index_b in range(index_a + 1, len(stats)):
			connectivity = determineConnectivity(img, stats[index_a][2], stats[index_a][1][::-1], stats[index_b][2], stats[index_b][1][::-1])

			if connectivity:
				NODE_INFO[index_a]['connections'].append(index_b)
				NODE_INFO[index_b]['connections'].append(index_a)
				cv2.line(display, stats[index_a][1][::-1], stats[index_b][1][::-1], [255,255,255], 4)

	return display, NODE_INFO

# if __name__ == '__main__':
# 	map_id = 'b185a918d35d4c04824ff829d337aad1'
# 	json_path = './json'

# 	img, info = generateAdjacency(*draw_map(map_id, json_path))

# 	cv2.imshow('Title', img)
# 	cv2.waitKey()
# 	cv2.imwrite('out.png', img)

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
		img, data = generateAdjacency(*draw_map(map_id, json_path))

		with open(save_path + "/" + map_id + '.json', 'w') as f:
			json.dump(data, f)
		
		cv2.imwrite(save_path + "/" + map_id + '.png', img)