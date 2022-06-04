from math import ceil
import os
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Img

L = 256
OutputDir = "output"
ALL_FILES = []
CURRENT_FILE = ""
IF_SHOW = False
Suffix = ".png"
USE_LIB = True

EnlargeEdge = 100
EnlargeArea = {'barb': [150, 220], 
			   'boat': [210, 300],
			   'lena': [180, 200],
			   'mandrill': [380, 230],
			   'peppers-bw': [210, 240]}

def show_pic(pic1: np.array, str1=''):
	global CURRENT_FILE

	if CURRENT_FILE not in EnlargeArea.keys():
		raise NotImplementedError

	ea = EnlargeArea[CURRENT_FILE]
	imsave(os.path.join(OutputDir, 'E_'+str1+'_'+CURRENT_FILE+'.png'), pic1[ea[0]:ea[0]+EnlargeEdge, ea[1]:ea[1]+EnlargeEdge], cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
	imsave(os.path.join(OutputDir, str1+'_'+CURRENT_FILE+'.png'), pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)

	if IF_SHOW:
		plt.imshow(pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
		plt.xticks([])
		plt.yticks([])
		plt.title(CURRENT_FILE)
		plt.show()

def map_point(FromShape, ToShape, Point):
	return (Point / (FromShape - [1, 1]) * (ToShape - [1, 1]))

def shrink_image(pic: np.array):
	H, W = pic.shape
	assert(H % 2 == 0)
	assert(W % 2 == 0)

	H, W = int(H/2), int(W/2)
	new_pic = np.zeros((H, W), dtype=int)
	for h in range(H):
		for w in range(W):
			new_h = h*2 if h < H/2 else h*2+1
			new_w = w*2 if w < W/2 else w*2+1
			new_pic[h, w] = pic[new_h, new_w]

	return new_pic

def NearestInterpolation(old_pic: np.array):
	H, W = old_pic.shape
	H, W = H*2, W*2
	if USE_LIB:
		new_pic = Img.fromarray(old_pic).resize((H, W), resample=Img.NEAREST)
		return np.array(new_pic)

	new_pic = np.zeros((H, W), dtype=np.int32)

	pic_shape, new_pic_shape = np.array(old_pic.shape), np.array(new_pic.shape)
	for h in range(H):
		for w in range(W):
			mapped_point = map_point(new_pic_shape, pic_shape, [h, w])
			mapped_point = round(mapped_point[0]), round(mapped_point[1])
			new_pic[h, w] = old_pic[mapped_point[0], mapped_point[1]]
		
	return new_pic

def BilinearInterpolation(old_pic: np.array):
	H, W = old_pic.shape
	H, W = H*2, W*2
	if USE_LIB:
		new_pic = Img.fromarray(old_pic).resize((H, W), resample=Img.BILINEAR)
		return np.array(new_pic)
	new_pic = np.zeros((H, W), dtype=np.int32)

	pic_shape, new_pic_shape = np.array(old_pic.shape), np.array(new_pic.shape)
	for h in range(H):
		for w in range(W):
			mapped_point = map_point(new_pic_shape, pic_shape, [h, w])
			h_map, w_map = mapped_point[0], mapped_point[1]
			h_min, w_min = int(mapped_point[0]), int(mapped_point[1])
			h_max, w_max = ceil(mapped_point[0]), ceil(mapped_point[1])

			h_min = 0 if h_min < 0 else h_min
			w_min = 0 if w_min < 0 else w_min
			h_max = H-1 if h_max > H-1 else h_max
			w_max = W-1 if w_max > W-1 else w_max

			assert(h_max-h_min in [0, 1])
			assert(w_max-w_min in [0, 1])

			s1 = (h_max-h_map) / (h_max-h_min) if h_max-h_min == 1 else 1
			s2 = (w_max-w_map) / (w_max-w_min) if w_max-w_min == 1 else 1

			f1 	= s1 * old_pic[h_min, w_min] + (1-s1) * old_pic[h_max, w_min]
			f2 	= (s1) * old_pic[h_min, w_max] + (1-s1) * old_pic[h_max, w_max]

			f 	= (s2) * f1 + (1-s2) * f2

			new_pic[h, w] = f

	return new_pic

def BicubicInterpolate(old_pic: np.array):
	H, W = old_pic.shape
	H, W = H*2, W*2
	if USE_LIB:
		new_pic = Img.fromarray(old_pic).resize((H, W), resample=Img.BICUBIC)
		return np.array(new_pic)
	new_pic = np.zeros((H, W), dtype=np.int32)

	pic_shape, new_pic_shape = np.array(old_pic.shape), np.array(new_pic.shape)
	for h in range(H):
		for w in range(W):
			mapped_point = map_point(new_pic_shape, pic_shape, [h, w])
			h_map, w_map = mapped_point[0], mapped_point[1]
			h_min, w_min = int(mapped_point[0]), int(mapped_point[1])
			h_max, w_max = ceil(mapped_point[0]), ceil(mapped_point[1])

			assert(h_max - h_min in [0, 1])
			assert(w_max - w_min in [0, 1])

			h_min = h_min-1 if h_max-h_min == 0 else h_min
			w_min = w_min-1 if w_max-w_min == 0 else w_min

			total_weights = 0
			total_value = 0
			for i in range(h_min-1,h_max+2):
				for j in range(w_min-1, w_max+2):
					if i < 0 or j < 0 or \
					   i >= old_pic.shape[0] or \
					   j >= old_pic.shape[1]:
						continue
					ww = BiCubic(i-h_map) * BiCubic(j-w_map)
					total_weights += ww
					total_value += (ww * old_pic[i, j])

			new_pic[h, w] = total_value / total_weights

	return new_pic

def BiCubic(x, alpha=-0.75):
	if x < 0:
		x = -x
	if x >= 2:
		return 0
	elif x <= 1:
		return (alpha+2) * np.power(x, 3) - (alpha+3)*np.power(x, 2) + 1
	else:
		return alpha*np.power(x, 3) - 5*alpha*np.power(x, 2) + 8*alpha*x - 4*alpha

def get_pic_loss(pic1, pic2):
	assert(pic1.shape == pic2.shape)
	mse = np.mean(np.power(pic1 - pic2, 2))
	psnr = 10 * np.log10(255 ** 2 / mse)
	return mse, psnr

def main():
	global CURRENT_FILE

	_, _, ALL_FILES = list(os.walk("./data"))[0]
	print(ALL_FILES)

	os.makedirs(OutputDir, exist_ok=True)
	print('USE_LIB=', USE_LIB)
	
	for file in ALL_FILES:
		CURRENT_FILE = os.path.splitext(os.path.basename(file))[0]

		pic = plt.imread(os.path.join("data", file))

		shrink_pic = shrink_image(pic)
		
		nearest_interpolate_pic = NearestInterpolation(shrink_pic)
		nearest_loss = get_pic_loss(pic, nearest_interpolate_pic)

		bilinear_interpolate_pic = BilinearInterpolation(shrink_pic)
		bilinear_loss = get_pic_loss(pic, bilinear_interpolate_pic)

		bicubic_interpolate_pic = BicubicInterpolate(shrink_pic)
		bicubic_loss = get_pic_loss(pic, bicubic_interpolate_pic)

		show_pic(pic, "Original")
		show_pic(nearest_interpolate_pic, "Nearest")
		show_pic(bilinear_interpolate_pic, "Bilinear")
		show_pic(bicubic_interpolate_pic, "Bicubic")

		print('-'*20 + CURRENT_FILE+ '-'*20)
		print('Nearest (MSE, PSNR): ', nearest_loss)
		print('Bilinear (MSE, PSNR): ', bilinear_loss)
		print('Bicubic (MSE, PSNR): ', bicubic_loss)

if __name__ == "__main__":
	main()
