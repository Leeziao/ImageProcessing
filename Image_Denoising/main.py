from copy import deepcopy
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
ShrinkWidth = 512
Sigma = 10
UseGridSearch = False
GradientDenoise = True

EnlargeEdge = 100
EnlargeArea = {'barb': [150, 220], 
			   'boat': [210, 300],
			   'lena': [180, 200],
			   'mandrill': [380, 230],
			   'peppers-bw': [210, 240]}

def show_pic(pic1: np.array, str1='', scale=True):
	global CURRENT_FILE

	pic1 = pic1.squeeze()

	filename = os.path.join(OutputDir, str(Sigma)+'_'+str1+'_'+CURRENT_FILE+'.png')
	e_filename = os.path.join(OutputDir, str(Sigma)+'_'+'E_'+str1+'_'+CURRENT_FILE+'.png')

	ea = EnlargeArea[CURRENT_FILE]
	ea_pic = pic1[ea[0]:ea[0]+EnlargeEdge, ea[1]:ea[1]+EnlargeEdge]

	if scale:
		imsave(filename, pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
		imsave(e_filename, ea_pic, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
	else:
		imsave(filename, pic1, cmap=plt.get_cmap('gray'))
		imsave(e_filename, ea_pic, cmap=plt.get_cmap('gray'))

	if IF_SHOW:
		if scale:
			plt.imshow(pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
			plt.imshow(ea_pic, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
		else:
			plt.imshow(pic1, cmap=plt.get_cmap('gray'))
			plt.imshow(ea_pic, cmap=plt.get_cmap('gray'))
		plt.xticks([])
		plt.yticks([])
		plt.title(CURRENT_FILE)
		plt.show()

def get_square_loss(pic: np.array, original_pic: np.array):
	grad = 2 * (pic - original_pic)
	loss = np.square(pic - original_pic).sum()
	return grad, loss


def get_sobolev_loss(pic: np.array, eps=1e-8):
	pic = pic.astype(np.double)
	x_diff = pic - np.concatenate((pic[:, 1:], pic[:, -1:]), axis=-1)
	y_diff = pic - np.concatenate((pic[1:, :], pic[-1:, :]), axis=0 )

	grad = np.square(x_diff) + np.square(y_diff)
	ret_loss = grad.sum()
	grad = np.sqrt(grad + eps)

	n_ux = x_diff / grad
	n_uy = y_diff / grad

	div_x = n_ux - np.concatenate((n_ux[:, :1], n_ux[:, :-1]), axis=-1)
	div_y = n_uy - np.concatenate((n_uy[:1, :], n_uy[:-1, :]), axis=0 )
	div = div_x + div_y

	grad_max = grad.max()
	if grad_max > 1:
		grad = grad / grad_max
	ret_grad = 2 * grad * div

	return ret_grad, ret_loss


def get_tv_loss(pic: np.array, eps=1e-8):
	pic = pic.astype(np.double)
	x_diff = pic - np.concatenate((pic[:, 1:], pic[:, -1:]), axis=-1)
	y_diff = pic - np.concatenate((pic[1:, :], pic[-1:, :]), axis=0 )

	grad = np.square(x_diff) + np.square(y_diff)
	grad = np.sqrt(grad + eps)
	loss = np.sum(grad)

	n_ux = x_diff / grad
	n_uy = y_diff / grad

	div_x = n_ux - np.concatenate((n_ux[:, :1], n_ux[:, :-1]), axis=-1)
	div_y = n_uy - np.concatenate((n_uy[:1, :], n_uy[:-1, :]), axis=0 )
	div = div_x + div_y

	return div, loss

def MAP_denoise(pic: np.array, reference:np.array, extra_loss_func=get_tv_loss, tau=1, lamb=1, iters=200, show_epoch=lambda x: x % 20 == 0, return_best=True):
	original_pic = deepcopy(pic)
	pic = pic.astype(np.double)

	momentum = np.zeros_like(pic).astype(np.double)
	momentum_beta = 0.9
	mse = []
	psnr = []
	best_mse = np.Inf
	best_pic = 0
	mse_iter, psnr_iter = 0, 0

	mse_iter, psnr_iter = get_pic_loss(pic, reference)
	mse.append(mse_iter)
	psnr.append(psnr_iter)

	for i in range(1, iters+1):
		extra_grad, extra_loss = extra_loss_func(pic)
		square_grad, square_loss = get_square_loss(pic, original_pic)
		# square_grad, square_loss = 0, 0

		grad = square_grad + lamb * extra_grad
		loss = square_loss + lamb * extra_loss

		# momentum *= momentum_beta
		# momentum += tau * grad * (1 - momentum_beta)
		# pic -= (momentum / (1 - momentum_beta ** i))
		pic -= tau * grad
		tau = tau * 0.9

		mse_iter, psnr_iter = get_pic_loss(pic, reference)
		mse.append(mse_iter)
		psnr.append(psnr_iter)
		if mse_iter < best_mse:
			best_mse, best_pic = mse_iter, deepcopy(pic)

		pic = clip_pic(pic)

		if show_epoch != None and show_epoch(i):
			print('Iter {}, Loss={}, MSE={}'.format(i, loss, mse_iter))
			# show_pic(pic, str(i))

	if return_best:
		return best_pic, np.array(mse), np.array(psnr)
	else:
		return pic, np.array([mse_iter]), np.array([psnr_iter])

def get_pic_loss(pic1, pic2):
	assert(pic1.shape == pic2.shape)
	pic1, pic2 = pic1.astype(np.double), pic2.astype(np.double)
	mse = np.mean(np.power(pic1 - pic2, 2))
	psnr = mse2psnr(mse)
	return mse, psnr

def mse2psnr(mse: np.array):
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr

def add_noise(pic, sigma=1):
	pic = pic.astype(np.float32)
	noise = np.random.randn(*pic.shape) * sigma
	pic = pic + noise
	pic = clip_pic(pic)
	return pic.astype(np.uint8), noise

def clip_pic(pic, low=0, high=255):
	return np.clip(pic, low, high)

def grid_search(tau_range, lamb_range):
	tt, ll = np.meshgrid(tau_range, lamb_range)
	t, l = tt.flatten(), ll.flatten()
	return list(zip(t, l))

def plot_grid_result(tau_range, lamb_range, result, caption:str=''):
	result = np.array(result).reshape(len(tau_range), len(lamb_range))
	tau_len, lamb_len = len(tau_range), len(lamb_range)
	tt, ll = np.arange(tau_len+1), np.arange(lamb_len+1)
	# tt, ll = np.meshgrid(tau_range, lamb_range)
	plt.pcolormesh(tt, ll, result, vmin=result.min(), vmax=result.max())
	plt.colorbar()
	plt.xlabel('tau')
	plt.ylabel('lambda')
	plt.xticks(np.arange(len(tau_range)) + 0.5, map(lambda x:str(round(x, 4)), tau_range))
	plt.yticks(np.arange(len(lamb_range)) + 0.5, map(lambda x:str(round(x, 4)), lamb_range))

	filename = os.path.join(OutputDir, caption+'_'+CURRENT_FILE+Suffix)
	plt.savefig(filename)

	if IF_SHOW:
		plt.show()
	plt.close()

def gradient_denoise():
	global CURRENT_FILE

	_, _, ALL_FILES = list(os.walk("./data"))[0]
	print(ALL_FILES)
	if UseGridSearch:
		print('#'*30, ' GridSearching ', '#'*30)

	os.makedirs(OutputDir, exist_ok=True)

	Pairs = [[get_tv_loss, 			'TV', 0.1, 10.0, 100],
			 [get_sobolev_loss, 	'Sobolev', 0.01, 100.0, 100]]
	# Pairs = [[get_tv_loss, 			'TV', 1, 1000.0],
	# 		 [get_sobolev_loss2, 	'Sobolev', 0.01, 1000.0]]
	tau_range = np.logspace(-3, 1, 5)
	lamb_range = np.logspace(-2, 2, 5)
	
	for file in ALL_FILES:
		CURRENT_FILE = os.path.splitext(os.path.basename(file))[0]

		pic = plt.imread(os.path.join("data", file))
		img = Img.fromarray(pic).resize((ShrinkWidth, ShrinkWidth), Img.NEAREST)
		pic = np.array(img)

		noise_pic, noise = add_noise(pic, sigma=Sigma)
		show_pic(noise, 'Noise', scale=False)
		n_mse, n_psnr = get_pic_loss(pic, noise_pic)

		for extra_f, extra_name, extra_tau, extra_lambda, extra_iter in Pairs:
			mse_grid = []
			psnr_grid = []

			packed_tl = [[1, 10]]
			if UseGridSearch:
				packed_tl = grid_search(tau_range, lamb_range)

			for tau, lamb in packed_tl:
				map_tv_pic, mse, psnr = MAP_denoise(noise_pic,
													pic,
													extra_loss_func=extra_f,
													iters=extra_iter,
													show_epoch=None,
													tau=tau if UseGridSearch else extra_tau,
													lamb=lamb if UseGridSearch else extra_lambda,
													return_best=False)
				best_index = np.argmin(mse)
				best_mse = mse[best_index]
				best_psnr = psnr[best_index]

				mse_grid.append(best_mse)
				psnr_grid.append(best_psnr)

				print('-'*20, CURRENT_FILE, '-'*20)
				print('[{}] Original v.s. Noise:    MSE({})  PSNR({})'.format(extra_name, round(n_mse, 2), round(n_psnr, 2)))
				print('[{}] Original v.s. Denoised.{}: MSE({})  PSNR({})'.format(extra_name, best_index, round(best_mse, 2), round(best_psnr, 2)))
				print()

				show_pic(pic,'Original')
				show_pic(noise_pic, 'Noised')
				show_pic(map_tv_pic, extra_name)

			if UseGridSearch:
				plot_grid_result(tau_range, lamb_range, mse_grid, 'MSE_'+extra_name)
				plot_grid_result(tau_range, lamb_range, psnr_grid, 'PSNR_'+extra_name)
		
		if UseGridSearch:
			exit(0)

def fourier_denoise():
	global CURRENT_FILE

	_, _, ALL_FILES = list(os.walk("./data"))[0]
	print(ALL_FILES)
	lamb = 1e-3

	for file in ALL_FILES:
		CURRENT_FILE = os.path.splitext(os.path.basename(file))[0]

		pic = plt.imread(os.path.join("data", file))
		img = Img.fromarray(pic).resize((ShrinkWidth, ShrinkWidth), Img.NEAREST)
		pic = np.array(img)

		noise_pic, _ = add_noise(pic, sigma=Sigma)

		f = np.fft.fft2(noise_pic)
		h, w = f.shape
		u_2 = np.repeat(np.array([[u * u for u in range(h)]]), w, axis=0)
		w_2 = np.transpose(u_2)
		d = 1 + lamb * (u_2 + w_2)

		nf = f / d
		npic = np.clip(np.abs(np.fft.ifft2(nf)), 0, 255)

		show_pic(npic.astype(np.uint8), "")

		n_mse, n_psnr = get_pic_loss(pic, noise_pic)
		d_mse, d_psnr = get_pic_loss(pic, npic)
		print('-'*20, CURRENT_FILE, '-'*20)
		print('Original v.s. Noise:    MSE({})  PSNR({})'.format(round(n_mse, 2), round(n_psnr, 2)))
		print('Original v.s. Denoised:    MSE({})  PSNR({})'.format(round(d_mse, 2), round(d_psnr, 2)))

def main():
	if GradientDenoise:
		print('\033[1;32m## Gradient Denoise ##\033[0m')
		gradient_denoise()
	else:
		print('\033[1;32m## Fourier Denoise ##\033[0m')
		fourier_denoise()

def helper():
	mse = np.array([69.08, 50.93, 51.39, 78.07, 54.91])
	mmse = np.mean(mse)
	psnr = mse2psnr(mse)
	mpsnr = np.mean(psnr)
	print('Mean MSE={}, Mean PSNR={}'.format(mmse, mpsnr))


if __name__ == "__main__":
	helper()
	# main()
