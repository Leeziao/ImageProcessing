import os
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Img

OutputDir = "output"
ALL_FILES = []
CURRENT_FILE = ""
IF_SHOW = False
Suffix = ".png"
ShrinkWidth = 128

def show_pic(pic1: np.array, str1=''):
	global CURRENT_FILE

	pic1 = pic1.squeeze()

	filename = os.path.join(OutputDir, str1+'_'+str(ShrinkWidth)+'_'+CURRENT_FILE+'.png')
	imsave(filename, pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)

	if IF_SHOW:
		plt.imshow(pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
		plt.xticks([])
		plt.yticks([])
		plt.title(CURRENT_FILE)
		plt.show()

# DFT
def dft(img):
	H, W, channel = img.shape
 
	G = np.zeros((H, W, channel), dtype=np.complex128)

	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)
 
	for c in range(channel):
		for v in range(H):
			for u in range(W):
				G[v, u, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * u / W + y * v / H)))
 
	return G

def show_dft(img):
	H, W, _ = img.shape
	ii, jj = np.meshgrid(np.arange(H), np.arange(W))

	new_img = img * np.power(-1, ii *jj)[:, :, np.newaxis]
	dft_img = dft(new_img).squeeze()
	dft_img = np.log(np.absolute(dft_img))

	filename = os.path.join(OutputDir, 'DFT_'+str(ShrinkWidth)+'_'+CURRENT_FILE+Suffix)
	plt.imsave(filename, dft_img, cmap=plt.get_cmap('gray'))

	if IF_SHOW:
		plt.imshow(dft_img, cmap=plt.get_cmap('gray'))
		plt.colorbar()
		plt.show()
 
# IDFT
def idft(G):
	H, W, channel = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)
 
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)
 
	for c in range(channel):
		for v in range(H):
			for u in range(W):
				out[v, u, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * u / W + y * v / H)))) / (H*W)
 
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)
 
	return out

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
	
	for file in ALL_FILES:
		CURRENT_FILE = os.path.splitext(os.path.basename(file))[0]

		pic = plt.imread(os.path.join("data", file))
		img = Img.fromarray(pic).resize((ShrinkWidth, ShrinkWidth), Img.NEAREST)
		pic = np.array(img)
		if pic.ndim == 2:
			pic = pic[:, :, np.newaxis]

		show_dft(pic)

		dft_pic = dft(pic)
		idft_pic = idft(dft_pic)

		mse, psnr = get_pic_loss(pic, idft_pic)

		show_pic(pic, "Original")
		show_pic(idft_pic, "IDFT")
		print('{}: MSE={}, PSNR={}'.format(CURRENT_FILE, mse, psnr))

if __name__ == "__main__":
	main()
