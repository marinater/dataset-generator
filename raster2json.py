import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage.measurements import center_of_mass
import sys

#Define constants for conversion
meter2pixel = 30
border_pad = 50
COMPLEMENTS = {
	"north": "south",
	"south": "north",
	"east": "west",
	"west": "east"
}
class ImageTiler:
	windows = []

	def __init__(self, name = 'Dataset Generator'):
		self.window_name = name

	def clear_tiles(self):
		cv2.destroyAllWindows()
		self.windows = []

	def add_tile(self, image):
		image = np.copy(image)

		if len(image.shape) == 2:
			image =  np.dstack((image, image, image))





		# if there are no windows or the last window is full
		# add
		if not self.windows or self.windows[-1].shape[1] + image.shape[1] > 1920:
			self.windows.append(image)
		else:
			print( self.windows[-1].shape, image.shape)
			self.windows[-1] = np.hstack(( self.windows[-1], image ))

		for index, window in enumerate(self.windows):
			cv2.imshow(str(index), window)

		[print(window.shape) for window in self.windows]

IMAGE_TILER = ImageTiler()

def draw_base_from_data(file_name, json_path):
	'''
	Draws image from JSON data. Modified version of code from HouseExpo documentation

	Input: map_id, json_path
	Outout: footprint, walls, json_data
	'''

	#Parse vertices from data and draw onto arrays
	#Footprint is a binary array with BACKGROUND in WHITE and HOUSE filled in BLACK
	#Color scheme is used later on so numpy |= operator can be used to efficiently transfer room colors
	#Walls is a binary array with only walls drawn in
	with open(json_path + '/' + file_name + '.json') as json_file:
		json_data = json.load(json_file)

	verts = (np.array(json_data['verts']) * meter2pixel).astype(np.int)
	boundaries = np.max(verts[:, 0]), np.min(verts[:, 0]),np.max(verts[:, 1]), np.min(verts[:, 1])
	x_max, x_min, y_max, y_min = boundaries

	footprint = np.full((y_max - y_min + border_pad * 2, x_max - x_min + border_pad * 2), 255, dtype=np.uint8)
	walls = np.zeros(footprint.shape, dtype=np.uint8)

	#Modify vertices to account for border padding and then draw
	verts[:, 0] = verts[:, 0] - x_min + border_pad
	verts[:, 1] = verts[:, 1] - y_min + border_pad
	cv2.drawContours(footprint, [verts], 0, 0, -1)
	cv2.drawContours(walls, [verts], 0, 255, 2)


	# IMAGE_TILER.add_tile(footprint)
	# IMAGE_TILER.add_tile(walls)

	return footprint, walls, json_data, boundaries

def draw_base_image(file_name, json_path):
	'''
	Creates image from vertex data and labels (colors) each room in with bounding box data
	Unknown rooms are labeled with 0
	Background is labeled 255

	Input: map_id, json_dir_path
	Output: annotated_image, known_room_count, known_room_bounding_boxes
	'''

	footprint, walls, json_data, boundaries = draw_base_from_data(file_name, json_path)
	x_max, x_min, y_max, y_min = boundaries

	#Create empty canvas to draw on and set starting room label to 1
	rooms_map = np.zeros(footprint.shape, dtype=np.uint8)
	fill_val = 1

	#Save bounding box info while iterating as well
	boxes = []

	for category, rooms in json_data['room_category'].items():
		for room in rooms:
			#Label each room and increment label for later
			bbox_tp = (np.array(room) * meter2pixel).astype(np.int)
			bbox = [np.max([bbox_tp[0] - x_min + border_pad, 0]),
					np.max([bbox_tp[1] - y_min + border_pad, 0]),
					np.min([bbox_tp[2] - x_min + border_pad, footprint.shape[1]]),
					np.min([bbox_tp[3] - y_min + border_pad, footprint.shape[0]])]

			boxes.append((category, bbox))
			rooms_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = fill_val
			fill_val += 1

	#Transfer the labels from rooms_map to footprint ONLY if the corresponding pixel on footprint is part of the house
	#Since bounding boxes are rectangular, this 'trims' the rooms to fit einside the house footprint
	footprint[footprint == 0] |= rooms_map[footprint == 0]

	#Transfer over the walls image to have the walls colored in more thickly
	footprint[walls == 255] = 255

	return footprint, fill_val, boxes

def label_unknown_rooms(rooms, fill_val):
	'''
	Labels regions of the footprint that the JSON data did not label

	Input: rooms_img, start_fill_value
	Output: fully_labeled_image, ending_fill_value
	'''

	#Create binary image with only unknown regions
	unknown = np.zeros(rooms.shape, dtype=np.uint8)
	unknown[rooms == 0] = 255

	#Label the connected regions and save the count as well
	num_labels, labels, _, _ = cv2.connectedComponentsWithStats(unknown)

	#Fill in the unlabeled regions
	for i in range(1, num_labels):
		region = np.nonzero(labels==i)
		coord = (region[0][0], region[1][0])[::-1]
		cv2.floodFill(rooms, np.pad(255 - unknown, 1, mode='constant'), coord , fill_val)
		fill_val += 1
	if np.any(rooms == 0): print('ERROR: Did not fill all unknown regions')

	return rooms

def determine_connectivity(rooms, pos_a, pos_b):
	'''
	Returns whether two regions are connected
	pos_a and pos_b are two points in the regions of interest

	Input: room_img, coordinates_a, coordinates_b
	Ouput: isConnected
	'''

	#Create two masks with just A and just B on them
	val_a = rooms[pos_a]
	val_b = rooms[pos_b]

	mask_a = np.zeros(rooms.shape, np.uint8)
	mask_a[rooms == val_a] = 255

	mask_b = np.zeros(rooms.shape, np.uint8)
	mask_b[rooms == val_b] = 255


	#Count the number of blobs in each mask
	mask_a_count, _ = cv2.connectedComponents(mask_a)
	mask_b_count, _ = cv2.connectedComponents(mask_b)

	#Count the number of blobs when both masks are combined
	combined_mask = mask_a + mask_b
	combined_count, _ = cv2.connectedComponents(combined_mask)

	#Regions are connected if two foreground blobs merge into one when
	#the masks are combined
	#Minus 1 to account for the extra background blob

	isConnected = combined_count < mask_b_count + mask_a_count - 1

	#Now that connectivity is properly determined, we can take the intersection
	#of the masks to isolate the door. Since we are dilating as part of this,
	#it was important to ensure that the roooms were originally connected
	#in the first place to prevent false positives

	if isConnected:
		mask_a = cv2.dilate(mask_a, np.ones((2,2)), iterations=1)
		mask_b = cv2.dilate(mask_b, np.ones((2,2)), iterations=1)

		door = (mask_a!=0) & (mask_b!=0).astype(np.uint8)

		rows = np.any(door, axis=1)
		cols = np.any(door, axis=0)
		ymin, ymax = np.where(rows)[0][[0, -1]]
		xmin, xmax = np.where(cols)[0][[0, -1]]

		isDoorVertical = ymax - ymin > xmax - xmin

		center_a = center_of_mass(mask_a)
		center_b = center_of_mass(mask_b)

		if isDoorVertical:
			if center_a[1] > center_b[1]: return True, 'east'
			else: return True, 'west'
		else:
			if pos_a[0] > pos_b[0]: return True, 'south'
			else: return True, 'north'

	else: return False, 'n/a'

def determine_label(boxes, pos):
	'''
	Returns corresponding label of coordinate from bounding box info

	Input: bounding_box_info, POI_coordinate,
	Output: label
	'''

	#Find the bounding box that contains the input coordinate and returns the label if there is any
	for label, box in boxes:
		if box[0] <= pos[1] + 3 <= box[2] and box[1] <= pos[0] + 3 <= box[3]:
			return label
	return 'Unknown'

def shift_coord(t, c = 5, d = 5):
	'''
	Reverses coordinate for opencv drawing purposes
	Also shifts to the bottom right for clarity
	'''
	return t[1] + c, t[0] + d

def h2rgb(img):
	'''
	Converts image to colorful image
	'''

	h = img.copy()
	h[img >= 244] = 0 #Remove lines and circles (drawn in 244 and 255) so normalizing 0-255 works as intended

	s = np.full(h.shape, 200, np.uint8)
	v = np.full(h.shape, 200, np.uint8)

	v[img == 254] = 0 #Set drawn in lines and circles to be white
	v[img == 255] = 100 #Set background to be grey
	s[img == 255] = 0 #Set background to be grey

	h = cv2.normalize(h, None, alpha=0, beta=179, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	hsv = np.dstack((h,s,v))

	return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def generate_data(rooms, boxes):
	'''
	Collect all data in one place, store, and display
	'''

	display = rooms.copy()
	coordinates = []

	for index, label_value in enumerate(np.unique(rooms)[:-1]):
		region = np.nonzero(rooms == label_value)

		coordinate = (region[0][0], region[1][0])
		coordinates.append(coordinate)

		cv2.circle(display, shift_coord(coordinate), 4, 254, 2)
		cv2.putText(display, str(index), shift_coord(coordinate, 10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, 254)
	data = [
		{
			'position': [d.item() for d in c],
			'square_footage': np.count_nonzero(rooms == rooms[c]),
			'connections': [],
			'label': determine_label(boxes, c)
		}
		for c in coordinates
	]

	# for d in data:
		# cv2.putText(display, d['label'], shift_coord(d['position'], 10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, 254)

	for a in range(len(coordinates) - 1):
		for b in range(a + 1, len(coordinates)):
			is_connected, orientation = determine_connectivity(rooms, coordinates[a], coordinates[b])

			if is_connected:
				cv2.line(display, shift_coord(coordinates[a]), shift_coord(coordinates[b]), 254, 2)
				data[a]['connections'].append({"node": b, "orientation": COMPLEMENTS[orientation]})
				data[b]['connections'].append({"node": a, "orientation": orientation})

	return h2rgb(display), data

if __name__ == '__main__':
	map_file = input('Map ID file (ENTER for default): ').strip()
	if map_file == '': map_file = './100.txt'
	elif map_file.isdigit(): map_file = './{0:}.txt'.format(map_file)

	json_path = './json'
	save_path = './out'

	map_ids = np.loadtxt(map_file, str)

	if not os.path.exists(save_path):
		os.mkdir(save_path)
# [130:131]
	for map_id in tqdm(map_ids):
		*draw_base_output, boxes = draw_base_image(map_id, json_path)
		labeled_rooms = label_unknown_rooms(*draw_base_output)
		display, data = generate_data(labeled_rooms, boxes)

		# IMAGE_TILER.add_tile(display)

		# cv2.waitKey()
		# IMAGE_TILER.clear_tiles()

		with open('./out/{0:}.json'.format(map_id), 'w') as outfile:
			json.dump(data, outfile)
			cv2.imwrite('./out/{0:}.png'.format(map_id), display)

# Image 130 in 35126.txt
