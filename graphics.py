import numpy as np
import torch
import cv2

# Global parameters:
#shifting_factor = 0.05
#sx, sy = 0.0133333000, 0.0100000000

class Graphics:
    def __init__(self, f, sx, sy, depth_levels, shifting_offset):
        self.f = f
        self.sx = sx
        self.sy = sy
        self.depth_levels = depth_levels
        self.shifting_offset = shifting_offset

    def plane_sweep(self, x_source, x_others, src_img, other_imgs):
        # input: other_imgs: (N, h, w, 3)
        N, h, w, _ = other_imgs.shape
        # f: focal length
        x_offsets = []
        n = np.array([0, 0, 1]).reshape(1, -1)

        R = np.eye(3)
        k = np.zeros((3, 3))
        k[0, 0] = self.f / self.sx
        k[1, 1] = self.f / self.sy
        k[2, 2] = 1
        px = 0
        py = 0
        k[0, 2] = px
        k[1, 2] = py
        k_inv = np.linalg.inv(k)

        for depth in self.depth_levels:
            a = -1 * depth
            x1 = x_others[0]
            single_space = x1 - x_source
            t_element = single_space * self.shifting_offset
            t = np.array([t_element, 0, 0]).reshape(-1, 1)
            denom = float(a - np.matmul(n, np.matmul(R.T, t)))
            numerat = np.matmul(R.T, np.matmul(t, np.matmul(n, R.T)))
            inv_hom = np.matmul(k, np.matmul(R.T + numerat / denom, k_inv))
            single_offset = inv_hom[0, 2]

            for x_other in x_others:
                x_offsets.append(round((x_other - x1) * single_offset))

        x_offsets = np.array(x_offsets).reshape(1, -1)

        x_others = np.array(x_others)
        x_offsets_concat0 = np.concatenate((x_offsets, np.zeros(len(x_offsets.flatten())).reshape(1, -1)), axis=0)
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

        for i in range(len(x_offsets.flatten())):
            shifted_images[i, :, output_fragment_1[i]:output_fragment_2[i], :] = other_imgs[i % len(x_others.flatten()), :, source_fragment_1[i]:source_fragment_2[i], :]
        shifted_images[-1] = src_img
        shifted_images = tuple(shifted_images)
        output_feats = np.concatenate(shifted_images, axis=2)
        return output_feats


    def translation_view_render(self, color_imgs, alpha_imgs, reference_x, target_x):
        #1: mask alpha images over the color images:
        # In the original source, the t represents the x in corrrespondce of the reference_x
        # output: novel_view: N, 3, h, w

        k = np.zeros((3, 3))
        k[0, 0] = self.f / self.sx
        k[1, 1] = self.f / self.sy
        k[2, 2] = 1
        px = 0
        py = 0
        k[0, 2] = px
        k[1, 2] = py

        h = alpha_imgs[0].shape[0]
        w = alpha_imgs[0].shape[1]

        # adjustable parameters:
        # depth max, depth_delta

        k[0, 2] = px
        k[1, 2] = py
        k_inv = np.linalg.inv(k)
        R = np.eye(3)
        t = np.array([(target_x - reference_x) * self.shifting_offset, 0, 0]).reshape(-1, 1)
        n = np.array([0, 0, 1]).reshape(1, -1)
        final_img = torch.zeros((3, h, w)).cuda()

        for l, depth in enumerate(self.depth_levels[::-1]):

            #depth = depth_max - l * depth_delta
            a = -1 * depth
            denom = float(a - np.matmul(n, np.matmul(R.T, t)))
            numerat = np.matmul(R.T, np.matmul(t, np.matmul(n, R.T)))
            inv_hom = np.matmul(k, np.matmul(R.T + numerat / denom, k_inv))

            x_offset = round(inv_hom[0, 2])
            layer_final_img = torch.zeros((3, h, w)).cuda()
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

    def get_mpi_singleinput(self, net_output, reference_x, reference_img):
        # return mpi tuple: (color images tensor, alpha images tuple)

        net_output = torch.squeeze(net_output)
        # net_output: size of (c, w, h)
        bk_color = net_output[:3]
        bl_weights = net_output[3:3 + len(self.depth_levels)]
        alpha_img = net_output[3+len(self.depth_levels):]

        num_layers = bl_weights.shape[0] # default 32
        h = net_output.shape[1]
        w = net_output.shape[2]

        color_imgs = torch.zeros((num_layers * 3, h, w)).cuda()
        for l in range(num_layers):
            # l*3, l*3+1, l*3+2
            color_imgs[l * 3:l * 3 + 3] = bk_color * bl_weights[l] + (1 - bl_weights[l]) * reference_img

        return (color_imgs, alpha_img, reference_x)

    def get_mpi_multinput(self, net_output, reference_x, reference_img):
        # expected output:
        # color_imgs: N, 3 * max_depth_level, h, w
        # alpha_imgs: N, max_depth_level, h, w

        # net_output: (N, c, h, w)
        # reference_x: list of reference xs
        bk_color = net_output[:, :3] #N, 3, h, w
        bl_weights = net_output[:, 3:3+len(self.depth_levels)] #N, 33, h, w
        alpha_img = net_output[:, 3+len(self.depth_levels):] #N, 33, h, w

        num_layers = bl_weights.shape[1]
        h = net_output.shape[2]
        w = net_output.shape[3]
        N = net_output.shape[0]

        color_imgs = torch.zeros((N, num_layers * 3, h, w)).cuda()
        for l in range(num_layers):
            # l*3, l*3+1, l*3+2
            color_imgs[:, l * 3:l * 3 + 3] = bk_color * bl_weights[:, l:l + 1, :, :] + (1 - bl_weights[:, l:l + 1, :, :]) * reference_img

        return (color_imgs, alpha_img, reference_x)

    # supports multiple input
    def get_visible_mpi(self, color_imgs, alpha_img):
        # input format:
        # color images: N, 3 * num of depth levels, h, w
        # alpha images: N, num of depth levels, h, w
        # input datatype: torch.Tensor
        transmittance_planes = []
        for i, depth in enumerate(self.depth_levels):
            alpha_initial = alpha_img[:, i:i+1, :, :] # output size: N, 1, h, w
            foreground_alpha = 1
            for j in range(i):
                foreground_alpha = foreground_alpha * (1 - alpha_img[:, j:j+1, :, :])
            transmittance_planes.append(alpha_initial * foreground_alpha) # format for the appended plane: N, 1, h, w

        transmittance_planes = torch.cat(transmittance_planes, 1)
        # get the visible mpi layer by layer:
        visible_mpi_colorimgs = []
        for i, depth in enumerate(self.depth_levels):
            visible_mpi_colorimgs.append(transmittance_planes[:, i:i+1, :, :] * color_imgs[:, i * 3: (i + 1) * 3, :, :]) # multiplying factor 1 shape: N, 1, h, w, multiplying factor 2 shape: N, 3, h, w
        visible_mpi_colorimgs = torch.cat(visible_mpi_colorimgs, 1) # output format: N, num planes * 3, h, w
        return visible_mpi_colorimgs

    def get_first_visible_imgs(self, visible_mpi_colorimgs):
        N, _, h, w = visible_mpi_colorimgs.shape
        first_visible_imgs = torch.zeros(N, 3, h, w).cuda()
        for i, depth in enumerate(self.depth_levels):
            first_visible_imgs = first_visible_imgs + visible_mpi_colorimgs[:, i * 3: (i + 1) * 3, :, :]
        return first_visible_imgs

    def get_mpi(self, color_imgs, alpha_imgs):
        # get the mpi representation images:
        for l, depth in enumerate(self.depth_levels):
            mpi_l = color_imgs[l * 3:(l + 1) * 3] * alpha_imgs[l]
            mpi_l = np.transpose((mpi_l * 255).clone().cpu().detach().numpy(), (1, 2, 0))
            cv2.imwrite('evaluate/mpi_' + str(depth) + '.png', mpi_l)

    def get_gray_mpi(self, color_imgs, alpha_imgs):
        # get the mpi representation images:
        mpis = []
        N, _, h, w = color_imgs.shape
        for l, depth in enumerate(self.depth_levels):
            mpis.append(torch.mean(color_imgs[:, l * 3:(l + 1) * 3, :, :] * alpha_imgs[:, l:l+1, :, :], 1).view((N, 1, h, w)))
        return torch.cat(mpis, 1) # Num_Depth, h, w

    def get_Pesudo_Mpi(self, reference_img, other_imgs):
        # calculate the original disparity map:
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        dismaps = []
        for i in range(len(other_imgs) - 1):
            dismaps.append(stereo.compute(cv2.cvtColor(np.asarray(reference_img * 255, dtype=np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(np.asarray(other_imgs[i] * 255, np.uint8), cv2.COLOR_BGR2GRAY)))
        disparity_map = np.mean(np.array(dismaps), 0)
        disparity_map = (disparity_map - min(disparity_map.reshape(-1))) / (max(disparity_map.reshape(-1)) - min(disparity_map.reshape(-1)))
        num_planes = len(self.depth_levels)
        depth_interval = 1 / num_planes
        discrete_disparity = disparity_map // depth_interval
        masked_planes = []
        for d in range(num_planes):
            mask = (discrete_disparity == d)
            h, w = mask.shape
            mask = mask.reshape(h, w, 1)
            masked_plane = mask * reference_img
            h, w, c = masked_plane.shape
            masked_plane = np.mean(masked_plane, 2)
            masked_planes.append(masked_plane)
        # masked planes shape: (num_planes, h, w)
        masked_planes = np.array(masked_planes)
        return masked_planes