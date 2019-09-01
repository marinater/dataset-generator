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
border_pad = 50

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

	orig = footprint.copy()
	rooms_map = np.full(footprint.shape, 0, dtype=np.uint8)
	fill_val = 1

	for category, rooms in json_data['room_category'].items():
		for room in rooms:
			bbox_tp = (np.array(room) * meter2pixel).astype(np.int)
			bbox = [np.max([bbox_tp[0] - x_min + border_pad, 0]),
					np.max([bbox_tp[1] - y_min + border_pad, 0]),
					np.min([bbox_tp[2] - x_min + border_pad, footprint.shape[1]]),
					np.min([bbox_tp[3] - y_min + border_pad, footprint.shape[0]])]

			rooms_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = fill_val
			fill_val += 1

	footprint[footprint == 0] |= rooms_map[footprint == 0]
	# footprint[walls == 255] = 255
	# unknown = np.zeros(footprint.shape, dtype=np.uint8)
	# unknown[footprint == 0] = 255

	# ret, markers = cv2.connectedComponents(unknown)
	# labels = np.unique(markers).nonzero()

	# out = np.zeros(footprint.shape, dtype=np.uint8)

	# for centroid in centroids:
	# 	unknown_centers.append(('Unknown', centroid, fill_val))
	# 	cv2.floodFill(out, np.pad(255 - unknown, 1, mode='constant'), centroid[::-1], fill_val)
	# 	fill_val += 1

	# for category, centroid, fill_val in centers:
	# 	mask = np.full(footprint.shape, 255, dtype=np.uint8)
	# 	mask[footprint == fill_val] = 0
	# 	cv2.floodFill(out, np.pad(mask, 1, mode='constant'), centroid[::-1], fill_val)

	# return np.concatenate((np.dstack((orig,orig,orig)), h2rgb(footprint)), axis=1)
	return np.concatenate((orig, footprint * 40), axis=1)
	# return out, centers + unknown_centers

if __name__ == '__main__':
	# map_id = '141e214ecf2a17992324ce713e3dd5d1'
	json_path = '../json'

	for filename in os.listdir(json_path)[:1000:10]:
		map_id = filename.replace('.json', '')
		cv2.imwrite('./out/' + map_id + '.png', draw_map(map_id, json_path))

	# cv2.imshow('Title', img)
	# cv2.waitKey()