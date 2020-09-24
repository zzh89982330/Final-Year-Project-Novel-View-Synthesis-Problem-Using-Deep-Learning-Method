import numpy as np
import torch

def plane_sweep(x_source, x_others, src_img, other_imgs, depth_levels, f):
    # input: other_imgs: (N, h, w, 3)
    N, h, w, _ = other_imgs.shape
    # f: focal length
    x_offsets = []
    n = np.array([0, 0, 1]).reshape(1, -1)
    
    R = np.eye(3)
    k = np.zeros((3, 3))
    k[0, 0] = f
    k[1, 1] = f
    k[2, 2] = 1
    px = src_img.shape[0] // 2
    py = src_img.shape[1] // 2
    k[0, 2] = px
    k[1, 2] = py
    k_inv = np.linalg.inv(k)
    
    for depth in depth_levels:
        a = -1 * depth
        for x in x_others:
            t_element = x - x_source
            t = np.array([t_element, 0, 0]).reshape(-1, 1)
            denom = float(a - np.matmul(n, np.matmul(R.T, t)))
            numerat = np.matmul(R.T, np.matmul(t, np.matmul(n, R.T)))
            inv_hom = np.matmul(k, np.matmul(R.T + numerat / denom, k_inv))
            final_img_hom_cor = np.zeros((h, w, 3))
            x_offset = round(inv_hom[0, 2])
            x_offsets.append(x_offset)
    
    x_offsets = np.array(x_offsets).reshape(1, -1)
    
    x_source = np.array(x_source)
    x_others = np.array(x_others)
    #x_offsets = (x_others - x_source).reshape(1, -1)
    x_offsets_concat0 = np.concatenate((x_offsets, np.zeros(len(x_offsets.flatten())).reshape(1, -1)), axis=0)
    #x_offsets_concatw = np.concatenate((x_offsets, np.ones(len(x_offsets)).reshape(1, -1) * w), axis=0)
    # max(0, x_offset):
    output_fragment_1 = np.max(x_offsets_concat0, axis=0).astype(int)
    # min(w, w + x_offset)
    output_fragment_2 = np.min(x_offsets_concat0 + w, axis=0).astype(int)
    # max(0, -x_offset)
    source_fragment_1 = np.max(x_offsets_concat0 * (-1), axis=0).astype(int)
    # min(w, w - x_offset)
    source_fragment_2 = np.min(x_offsets_concat0 * (-1) + w, axis=0).astype(int)
    # output fragment:
    shifted_images = np.zeros((len(x_offsets.flatten()) + 1, h, w, 3))
    #shifted_images[:, :, output_fragment_1:output_fragment_2, :] = other_imgs[:, :, source_fragment_1:source_fragment_2, :]
    
    for i in range(len(x_offsets.flatten())):
        shifted_images[i, :, output_fragment_1[i]:output_fragment_2[i], :] = other_imgs[i % len(x_others.flatten()), :, source_fragment_1[i]:source_fragment_2[i], :]
    shifted_images[-1] = src_img
    shifted_images = tuple(shifted_images)
    output_feats = np.concatenate(shifted_images, axis=2)
    return output_feats

def translation_view_render(color_imgs, alpha_imgs, reference_x, target_x):
    #1: mask alpha images over the color images:
    # In the original source, the t represents the x in corrrespondce of the reference_x
    
    k = np.zeros((3, 3))
    f = 5
    k[0, 0] = f
    k[1, 1] = f
    k[2, 2] = 1
    h = alpha_imgs[0].shape[0]
    w = alpha_imgs[0].shape[1]
    px = h // 2
    py = w // 2
    
    # adjustable parameters:
    # depth max, depth_delta
    depth_delta = 10
    depth_max = 50
    # depth list default with minimum of 1 and maximum of 100:
    depth_list = list(np.arange(1, 99, 98 / 31))
    
    k[0, 2] = px
    k[1, 2] = py
    k_inv = np.linalg.inv(k)
    R = np.eye(3)
    t = np.array([target_x - reference_x, 0, 0]).reshape(-1, 1)
    n = np.array([0, 0, 1]).reshape(1, -1)
    final_img = torch.zeros((3, h, w))#.cuda()
    
    # number of mpi layers:
    num_layers = alpha_imgs.shape[0]

    for l, depth in enumerate(depth_list):
        
        #depth = depth_max - l * depth_delta
        a = -1 * depth
        denom = float(a - np.matmul(n, np.matmul(R.T, t)))
        numerat = np.matmul(R.T, np.matmul(t, np.matmul(n, R.T)))
        inv_hom = np.matmul(k, np.matmul(R.T + numerat / denom, k_inv))
        final_img_hom_cor = np.zeros((h, w, 3))
        
        y = np.linspace(h - 1, 0, h)
        x = np.linspace(0, w - 1, w)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten()
        yv = yv.flatten()
        zv = np.ones((h, w)).flatten()
        final_img_hom_cor = np.concatenate((xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)), axis=1).reshape(h, w, 3)
        x_offset = round(inv_hom[0, 2])
        layer_final_img = torch.zeros((3, h, w))#.cuda()
        output_fragment = (max(0, x_offset), min(w, w + x_offset))
        source_fragment = (max(0, -x_offset), min(w, w - x_offset))
                        
        layer_final_img[:, :, int(output_fragment[0]):int(output_fragment[1])] = color_imgs[l * 3: (l + 1) * 3, :, int(source_fragment[0]):int(source_fragment[1])]
        final_img = final_img * (1 - alpha_imgs[l]) + layer_final_img * alpha_imgs[l]
        
    return final_img

"""
Net output:
(shape): 
background color: 3
blending weights: 32
alpha images: 32
"""

def get_mpi_singleinput(net_output, reference_x):
    # return mpi tuple: (color images tensor, alpha images tuple)
    
    net_output = torch.squeeze(net_output)
    # net_output: size of (c, w, h)
    bk_color = net_output[:3]
    bl_weights = net_output[3:36]
    alpha_img = net_output[36:]
    
    num_layers = bl_weights.shape[0] # default 32
    h = net_output.shape[1]
    w = net_output.shape[2]
    
    color_imgs = torch.zeros((num_layers * 3, h, w))
    for l in range(num_layers):
        # l*3, l*3+1, l*3+2
        color_imgs[l * 3:l * 3 + 3] = bk_color * bl_weights[l]
    
    return (color_imgs, alpha_img, reference_x)


