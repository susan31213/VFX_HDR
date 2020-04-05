import cv2
import imageio
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
import sys

np.warnings.filterwarnings('ignore')


img_dir = 'Images/4_aligned/'
list_file = 'image_list.txt'
LAMDBA = 50

def progress(count, total, status):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ... %s \r' % (bar, percents, '%', status))
    sys.stdout.flush()


def load_img(dir, rgb):
    img = []
    for f in dir:
        img.append(cv2.imread(img_dir + f)[:,:,rgb])
    return img


def response_curve_Debevec(imgs, exposures):
    Z = []
    B = []
    lamdba = LAMDBA
    w = []
    for img in imgs:
        Z.append(cv2.resize(img,(5,5)).flatten())
    for e in exposures:
        B.append(math.log(e))
    for i in range(256):
        w.append(i if i <= 255/2 else 255-i)

    n = 256
    A = np.zeros(shape=(np.size(Z, 1) * np.size(Z, 0) + n + 1, n + np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # fill in A with Zij & wij & tj
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            zij = Z[j][i]
            wij = w[zij]
            A[k][zij] = wij
            A[k][n+i] = -wij
            b[k] = wij * B[j]
            k += 1

    # fix g(128) = 0
    A[k][127] = 1
    k += 1

    # evaluate smoothness by g''
    for i in range(n-1):
        A[k][i] = lamdba * w[i+1]
        A[k][i+1] = -2 * lamdba * w[i+1]
        A[k][i+2] = lamdba * w[i+1]
        k += 1

    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lnE = x[256:]

    return g, lnE


def response_curve_Mitsunaga(imgs, exposures):
    Z = []
    B = []
    for img in imgs:
        Z.append(cv2.resize(img,(5,5)).flatten())

    # normalize pixel values to [0,1]
    M = [[i / 255 for i in z] for z in Z]
    # P: number of pixel, Q: number of exposures
    P = len(M[0])
    Q = len(M)
    # max order
    N = 10
    # stopping condition
    epsilon = 0.001
    max_iter = 10
    Imax = 200

    # coefficients matrix
    C = np.zeros((N, N+1))
    # let ratios be in increasing order
    M.reverse()
    B = exposures.copy()
    B.reverse()

    min_error = np.Inf;
    min_N = 0;

    for i in range(N):
        # print('order %d' % i)
        # ratio vector
        R = np.zeros((len(B)-1,1))
        for j in range(len(B)-1):
            R[j] = B[j]/B[j+1]
        # I matrix
        I = np.zeros((P,Q-1))
        I_last = np.zeros(I.shape)
        A = np.zeros((P*(Q-1)+1,i+1))
        b = np.zeros((A.shape[0],1))
        b[-1] = Imax
        cs = np.zeros((i+1,1))

        for k in range(max_iter):
            # print('iter %d' %k)
            for p in range(P):
                for q in range(Q-1):
                    j = p*(Q-1) + q
                    for n in range(i+1):
                        A[j][n] = math.pow(M[q][p], n) - R[q]*math.pow(M[q+1][p], n)
            A[-1] = np.ones((1, i+1))
            # compute coefficients by solving linear system
            cs = linear_model.Ridge(alpha=10)
            cs.fit(A, b)
            # print(np.hstack((cs.intercept_, cs.coef_[0])))
            # recompute ratios
            for q in range(Q-1):
                acc = 0
                for p in range(P):
                    num = cs.intercept_
                    den = cs.intercept_
                    for n in range(1, i+1):
                        num += cs.coef_[0][n-1]*math.pow(M[q][p], n)
                        den += cs.coef_[0][n-1]*math.pow(M[q+1][p], n)
                    acc = acc + num/den
                R[q] = acc
            # compute f for early stopping
            for p in range(P):
                for q in range(Q-1):
                    acc = cs.intercept_
                    for n in range(1, i+1):
                        acc += cs.coef_[0][n-1]*math.pow(M[q][p], n)
                    I[p][q] = acc
            if np.amax((I-I_last)) < epsilon:
                break
        #  compute error for final coefficients to decide N
        C[i][0:i+2] = np.hstack((cs.intercept_, cs.coef_[0]))
        error = 0
        for p in range(P):
            for q in range(Q-1):
                acc = cs.intercept_
                for n in range(1, i+1):
                    acc += cs.coef_[0][n-1]*(math.pow(M[q][p], n) - R[q]*math.pow(M[q+1][p], n))
                error += math.pow(acc, 2)
        if min_error >= error:
            min_error = error
            min_N = i
    c = C[min_N]
    n = 256
    g = np.zeros((n,1))
    for j in range(n):
        f = 0
        for i in range(N+1):
            f += c[i]*math.pow((j/(n-1)),i)
        g[j] = f
    return g


def radiance_map(g, imgs, ln_t, w, status):
    Z = np.array([img.flatten() for img in imgs])
    acc_E = np.zeros(Z[0].shape)
    ln_E = np.zeros(Z[0].shape)
    
    pixels, imgs = Z.shape[1], Z.shape[0]
    for i in range(pixels):
        progress(i, pixels, status)
        acc_w = 0
        for j in range(imgs):
            z = Z[j][i]
            acc_E[i] += w[z]*(g[z] - ln_t[j])
            acc_w += w[z]
        ln_E[i] = acc_E[i]/acc_w if acc_w > 0 else 0
    
    return ln_E


def construct_hdr(img_list, curve_list, exposures):
    img_size = img_list[0][0].shape
    w = []
    for i in range(256):
        w.append(i if i < 127.5 else 255-i)

    hdr = np.zeros((img_size[0], img_size[1], 3), 'float32')
    vexp = np.vectorize(lambda x:math.exp(x))
    # construct RGB radiance map
    for i in range(3):
        # print("  {0} channel ... ".format('b' if i == 0 else ('g' if i == 1 else 'r')))
        E = radiance_map(curve_list[i], img_list[i], np.log(exposures), w, 'b channel' if i == 0 else ('g channel' if i == 1 else 'r channel'))
        hdr[..., i] = np.reshape(vexp(E), img_size)     # exponational RGB channels and reshape to image size
        print('')
    return hdr


# Code from https://github.com/SSARCandy/HDR-imaging/blob/master/HDR-playground.py
def save_hdr(hdr, filename):
    image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
    image[..., 0] = hdr[..., 2]
    image[..., 1] = hdr[..., 1]
    image[..., 2] = hdr[..., 0]

    # cv2.imwrite('test_cv.hdr', hdr)

    f = open(filename, 'wb')
    f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1]) 
    f.write(bytes(header, encoding='utf-8'))

    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()


def tone_mapping(hdr, method='global', d=1e-5, a=0.5):
    if method == 'global':
        Lw = hdr
        Lw_bar = np.exp(np.mean(np.log(d + Lw)))
        Lm = (a / Lw_bar) * Lw
        Lm_white = np.max(Lm)
        Ld = (Lm * (1 + (Lm / (Lm_white ** 2)))) / (1 + Lm)
        ldr = np.clip(np.array(Ld * 255), 0, 255)
        return ldr.astype(np.uint8)

    elif method == 'local':
        return 
    else:
        print('No this tone mapping method...')
        return

if __name__ == '__main__':
    
    # Read list and get file paths
    f = open(img_dir + list_file, 'r')
    list_lines = f.readlines()
    files = []
    exposures = []
    for line in list_lines:
        s = line.split()
        files.append(s[0])
        exposures.append(float(s[1]))
    
    # Read img
    print("Loading images ... ")
    imgs_r = load_img(files, 2)
    imgs_g = load_img(files, 1)
    imgs_b = load_img(files, 0)

    # Recovering response curve by Mitsunaga's method
    print("Mitsunaga's method ... ")
    g_r_m = response_curve_Mitsunaga(imgs_r, exposures)
    g_g_m = response_curve_Mitsunaga(imgs_g, exposures)
    g_b_m = response_curve_Mitsunaga(imgs_b, exposures)

    # plt.figure(figsize=(10, 10))
    # plt.plot([math.log(i+0.000001) for i in g_r_m], range(256), 'r')
    # plt.plot([math.log(i+0.000001) for i in g_g_m], range(256), 'g')
    # plt.plot([math.log(i+0.000001) for i in g_b_m], range(256), 'b')
    # plt.ylabel('Pixel value')
    # plt.xlabel(r"$\ln(E_{i}) + \ln(\Delta t_{j})$")
    # plt.savefig('response-curve_m.png')

    print(" Construct hdr ... ")
    hdr_m = construct_hdr([imgs_b, imgs_g, imgs_r], [g_b_m, g_g_m, g_r_m], exposures)

    # Save Radiance map
    print(" Save Radiance map ... ", end='')
    plt.figure(figsize=(12,8))
    plt.imshow(np.log(cv2.cvtColor(hdr_m, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig(img_dir+'radiance-map_m.png')
    print('ok')

    # Save hdr file
    print(" Save hdr file ... ", end='')
    save_hdr(hdr_m, img_dir+'memorial_m.hdr')
    print('ok')

    # tone mapping
    print(" Tone mapping...", end='')
    global_ldr = tone_mapping(hdr_m, 'global')
    cv2.imwrite(img_dir+'global_ldr_m.jpg', global_ldr)
    print('ok')

    
    # Recovering response curve by Debevec's method
    print("Debevec's method ... ")
    g_r, lnE_r = response_curve_Debevec(imgs_r, exposures)
    g_g, lnE_g = response_curve_Debevec(imgs_g, exposures)
    g_b, lnE_b = response_curve_Debevec(imgs_b, exposures)

    # plt.figure(figsize=(10, 10))
    # plt.plot(g_r, range(256), 'r')
    # plt.plot(g_g, range(256), 'g')
    # plt.plot(g_b, range(256), 'b')
    # plt.ylabel('Pixel value')
    # plt.xlabel(r"$\ln(E_{i}) + \ln(\Delta t_{j})$")
    # plt.savefig('response-curve.png')
    
    # Reconstruct hdr
    print(" Construct hdr ... ")
    hdr = construct_hdr([imgs_b, imgs_g, imgs_r], [g_b, g_g, g_r], exposures)
    

    # Save Radiance map
    print(" Save Radiance map ... ", end='')
    plt.figure(figsize=(12,8))
    plt.imshow(np.log(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig(img_dir+'radiance-map.png')
    print('ok')

    print(" Save hdr file ... ", end='')
    save_hdr(hdr,img_dir+'memorial.hdr')
    print('ok')

    # tone mapping
    print(" Tone mapping...", end='')
    global_ldr = tone_mapping(hdr, 'global')
    cv2.imwrite(img_dir+'global_ldr.jpg', global_ldr)
    print('ok')