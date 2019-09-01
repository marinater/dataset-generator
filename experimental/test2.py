import numpy as np
import cv2
from matplotlib import pyplot as plt
from test import draw_map

map_id = 'b185a918d35d4c04824ff829d337aad1'
json_path = './json'

img = draw_map(map_id, json_path)

# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)

# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers+1
# markers[unknown==255] = 0

# out = np.zeros((*sure_bg.shape, 3), dtype=np.uint8)

# markers = cv2.watershed(out, markers)
# out[sure_bg > 0 ] = [255,255,255]
# out[markers == -1] = [44,252,3]


cv2.imshow('out', rgb)
cv2.imwrite('out.png', rgb)
cv2.waitKey()