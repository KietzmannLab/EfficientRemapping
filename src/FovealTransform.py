import numpy as np
import torch
import torchvision



class FovealTransform(torch.nn.Module):
    def __init__(self, fovea_size, img_target_size, img_size, jitter_type, jitter_amount, device, warp_imgs=True, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.roh_0 = int(fovea_size * (img_size[1] // 2))
        self.roh_max = int(img_size[1] // 2)
        retina_resolution = img_target_size if img_size[0] == 256 else img_target_size * 2

        self.N_r, self.resulting_resolution = self.find_number_of_rings(retina_resolution, self.roh_max, self.roh_0)
        self.N_r = int(self.N_r)
        self.x_0 = int(img_size[0] // 2)
        self.y_0 = int(img_size[1] // 2)
        self.img_target_size = img_target_size
        self.img_size = img_size
        self.jitter_type = jitter_type
        self.jitter_amount = int(jitter_amount)
        if warp_imgs:
            self.retina_coordinates, self.fovea_mask = self.create_sampling_coordinates(self.N_r,
                                                                                        self.roh_0,
                                                                                        self.roh_max,
                                                                                        self.x_0,
                                                                                        self.y_0)

            # postprocessing of coordinates to get rid of artifacts
            while self.retina_coordinates[0, -1][0] <= self.retina_coordinates[0, -5][0]:
                self.retina_coordinates = self.retina_coordinates[1:-1, 1:-1, :]
                self.fovea_mask = self.fovea_mask[1:-1, 1:-1, :]

            self.retina_coordinates[:, :, 0] = (self.retina_coordinates[:, :, 0] * 2 - img_size[1]) / img_size[1]  # between -1 and 1 because gridsampler requires that
            self.retina_coordinates[:, :, 1] = (self.retina_coordinates[:, :, 1] * 2 - img_size[0]) / img_size[0]
            self.retina_coordinates = self.retina_coordinates.to(self.device)
        else:
            self.retina_coordinates = self.create_crop_coordinates()
            self.retina_coordinates = self.retina_coordinates.to(self.device)

    def get_fcg_coordinates(self, x, y, N_r, roh_0, roh_max, x_0, y_0, inverse=True):
        """
        Given a coordinate pair (x,y) in the original image,
        this function computes the foveal cartesian geometry coordinates mapped back to
        cartesian coordinates on the image, given a number of rings N_r, the half width of the fovea roh_0,
        the half-size of the image roh_max, and the image center x_0, y_0).

        """
        self.x_0 = x_0
        self.y_0 = y_0
        # Preparation
        self.a = np.exp((np.log(roh_max / roh_0) / N_r))
        self.r = 0
        self.Gamma = np.empty((self.N_r))
        self.Gamma[0] = self.roh_0

        self.roh = np.empty((self.N_r))
        self.roh[0] = self.roh_0
        self.Ring = np.empty((self.N_r))
        self.Ring[0] = self.r
        for i in range(1, self.N_r):
            self.roh[i] = np.floor(self.roh_0 * self.a ** i)
            if (self.roh[i] < self.Gamma[self.r]) or (self.roh[i] > self.Gamma[self.r]):
                self.r = self.r + 1
                self.Gamma[self.r] = self.roh[i]
            self.Ring[i] = self.r
        self.foveaHeight = self.foveaWidth = 2 * (self.roh_0 + self.N_r - 1) + 1
        self.fcx = self.fcy = np.floor(self.foveaWidth / 2)

        def mapping(x, y):
            x = x - self.x_0
            y = y - self.y_0
            r = max(np.abs(x), np.abs(y))
            if r <= roh_0:
                x_prime = x + self.fcx
                y_prime = y + self.fcy
            else:
                xi = int(np.floor(np.log(r / roh_0) / np.log(self.a)))
                xi = min(xi, len(self.Ring) - 1)
                delta_roh_xi = (roh_0 + self.Ring[xi]) / (np.floor(roh_0 * self.a ** xi))
                x_prime = np.floor(x * delta_roh_xi + self.fcx)
                y_prime = np.floor(y * delta_roh_xi + self.fcy)
            return x_prime, y_prime

        def inverse_mapping(x_prime, y_prime):
            x_prime = x_prime - self.fcx
            y_prime = y_prime - self.fcy
            roh = max(np.abs(x_prime), np.abs(y_prime))
            if roh <= self.roh_0:
                x = x_prime + self.x_0
                y = y_prime + self.y_0
            else:
                index = int(roh - self.roh_0)
                index = min(index, len(self.Gamma) - 1)
                delta_roh_Gamma = self.Gamma[index] / roh
                x = np.floor(x_prime * delta_roh_Gamma + self.x_0)
                y = np.floor(y_prime * delta_roh_Gamma + self.y_0)
            return x, y

        if inverse:
            x, y = inverse_mapping(x, y)

            return (x, y)
        else:
            x_prime, y_prime = mapping(x, y)
            return (x_prime, y_prime)

    def create_sampling_coordinates(self, N_r, roh_0, roh_max, x_0, y_0):
        """
        Given a number of rings N_r, a fovea radius roh_0, an image radius roh_max, and central fixation coordinates
        x_0 and y_0 (==roh_max)
        """
        assert y_0 == roh_max, "fixation coordinates are expected to be in the center"

        self.a = np.exp((np.log(self.roh_max / self.roh_0) / self.N_r))
        self.max_delta_roh = (self.roh_0 + self.N_r) / np.floor((self.a ** (self.N_r - 1)) * self.roh_0)
        self.fcx = self.fcy = np.floor((2 * (self.roh_0 + self.N_r) + 1) / 2)
        self.max_x_prime = int(np.floor(self.roh_max * self.max_delta_roh + self.fcx))
        self.max_y_prime = int(np.floor(self.roh_max * self.max_delta_roh + self.fcy))

        new_coordinates = np.empty((self.max_x_prime, self.max_y_prime, 2))
        for i in range(0, self.max_x_prime):
            for j in range(0, self.max_y_prime):
                x, y = self.get_fcg_coordinates(i, j, N_r, roh_0, roh_max, x_0, y_0, inverse=True)
                new_coordinates[i, j, 0] = x
                new_coordinates[i, j, 1] = y

        # create a fovea mask (with ones where the fovea is not), used to add irregularity to peripheral cone locations
        fovea_mask = np.ones((self.max_x_prime, self.max_y_prime, 2))
        fovea_mask[self.max_x_prime // 2 - self.roh_0:self.max_x_prime // 2 + self.roh_0,
        self.max_y_prime // 2 - self.roh_0:self.max_y_prime // 2 + self.roh_0, :] = 0.
        fovea_mask = torch.Tensor(fovea_mask)
        return torch.Tensor(new_coordinates[:, :, ::-1].astype(np.float32)), fovea_mask

    def create_crop_coordinates(self):
        """
        Create grid coordinates for an image crop based on img_size and img_target_size
        """
        range = self.img_target_size / self.img_size[0]
        crop_coordinates = torch.ones(self.img_target_size, self.img_target_size, 2)
        crop_coordinates[:, :, 0] = torch.linspace(-1 * range, 1 * range, self.img_target_size).repeat(self.img_target_size, 1)
        crop_coordinates[:, :, 1] = torch.linspace(-1 * range, 1 * range, self.img_target_size).repeat(self.img_target_size, 1).T
        return crop_coordinates

    # @tf.function(jit_compile=True)
    def add_jitter(self, retina_warp_coordinates, fovea_mask, jitter_amount=0.0, jitter_type="gaussian"):
        """
        adds uniform jitter to peripheral area to imitate the irregular sampling properties of the retina.

        Expects the unshifted warping coordinates and a fovea mask with ones everywhere except in the central fovea.
        """
        keep_identical_mask = fovea_mask.int()
        if jitter_type == "uniform":
            jitter_amount = int(jitter_amount)
            jitter = torch.round(
                (torch.rand(retina_warp_coordinates.shape) * 2 - 1) * jitter_amount) * keep_identical_mask
            jitter = jitter * 2 / self.img_size[0]
        elif jitter_type == "gaussian":
            jitter = (torch.randn(
                retina_warp_coordinates.shape) * jitter_amount * keep_identical_mask.float()).int()  # /self.img_size
            jitter = jitter * 2 / self.img_size[0]
            # jitter = tf.cast(tf.random.normal(shape=tf.shape(retina_warp_coordinates),
            #                          mean=0.0, stddev=jitter_amount) * tf.cast(keep_identical_mask,tf.float32), tf.int32)

        return retina_warp_coordinates + jitter

    def warp_images(self, images, fixations, retina_warp_coordinates,
                    jitter_amount=0.0, jitter_type="gaussian"):
        """
        Takes a batch of (a time series of) images and a batch of (a sequence of) fixations, as well as the
        retina warp coordinates, pointing to locations on the original image (or beyond, in which case zeros are sampled),
        and samples a foveal transformed batch of image(s) from the original image(s).

        retina_warp_coordinates is expected to not have a batch dimension, so (x_dim, y_dim, 2)
        """
        # TODO cpu or cuda?
        fixations = fixations.float().cpu()
        batch_size = images.shape[0]
        # merge batch and time dimension
        if len(fixations.shape) > 2:
            # if len(tf.shape(fixations))>2:
            # if there is only one image but a sequence of fixations, repeat the image
            if not len(images.shape) > 4:
                if images.shape[1] != 3:
                    images = torch.unsqueeze(images, axis=1).repeat(1, 1, fixations.shape[1], 1, 1)  # B, C, T, H, W
                else:
                    images = torch.unsqueeze(images, axis=2).repeat(1, 1, fixations.shape[1], 1, 1)  # B, C, T, H, W
            time_steps = images.shape[2]
            height = images.shape[3]
            width = images.shape[4]
            images = torch.reshape(images, (batch_size * time_steps, images.shape[1], height, width))
            no_time_dim = False
        else:
            height = images.shape[1]
            width = images.shape[2]
            time_steps = 1
            no_time_dim = True

        # match batch+time dimension for retina warp coordinates
        retina_warp_coordinates = retina_warp_coordinates.repeat(batch_size * time_steps, 1, 1, 1)  # B*T, h,w,2

        # add jitter while still assuming central fixation
        if jitter_amount > 0.0:
            retina_warp_coordinates = self.add_jitter(retina_warp_coordinates,
                                                      self.fovea_mask,
                                                      self.jitter_amount,
                                                      self.jitter_type)
        if not no_time_dim:
            fixations = torch.reshape(fixations, (batch_size * time_steps, 2))
        x = fixations[:, 0].unsqueeze(1).unsqueeze(1).to(device=self.device)
        y = fixations[:, 1].unsqueeze(1).unsqueeze(1).to(device=self.device)
        # x = fixations[:, 0].unsqueeze(1).unsqueeze(1)
        # y = fixations[:, 1].unsqueeze(1).unsqueeze(1)
        shifted_retina_warp_coordinates = retina_warp_coordinates
        shifted_retina_warp_coordinates[:, :, :, 0] = retina_warp_coordinates[:, :, :, 0] + x
        shifted_retina_warp_coordinates[:, :, :, 1] = retina_warp_coordinates[:, :, :, 1] + y

        # fixations = fixations.repeat(1, retina_warp_coordinates.shape[1], retina_warp_coordinates.shape[2], 1,1).reshape((retina_warp_coordinates.shape[0],retina_warp_coordinates.shape[1],retina_warp_coordinates.shape[2], 2)) # b, h, w, 2

        # fixations = fixations - height//2
        # dx = torch.unsqueeze(torch.unsqueeze((x - float(height//2)),dim=1),dim=1)
        # dy = torch.unsqueeze(torch.unsqueeze((y - float(width//2)),dim=1),dim=1)
        # shifted_retina_warp_coordinates = retina_warp_coordinates + fixations
        # TODO why on cpu? does cuda also work?
        images = images.to(self.device)
        shifted_retina_warp_coordinates = shifted_retina_warp_coordinates.to(self.device)
        # print(shifted_retina_warp_coordinates.shape, shifted_retina_warp_coordinates.max().item(), shifted_retina_warp_coordinates.min().item(), shifted_retina_warp_coordinates[:, :, 0].mean().item(), shifted_retina_warp_coordinates[:, :, 1].mean().item())
        
        # TODO align corners?
        warped_images = torch.nn.functional.grid_sample(input=images, grid=shifted_retina_warp_coordinates, align_corners=False).to(
            self.device)  # tfa.image.resampler(data=images, warp=shifted_retina_warp_coordinates)
        # plt.imshow(warped_images.numpy()[0, 0, :, :])
        # plt.show()

        if no_time_dim:
            return warped_images

        else:
            # restore time dimension from merged batch*time dimension
            return torch.reshape(warped_images, (batch_size,
                                                 warped_images.shape[-3],
                                                 time_steps,
                                                 warped_images.shape[-2],
                                                 warped_images.shape[-1]))
        
    

    def find_number_of_rings(self, desired_resolution, roh_max, roh_0):
        """
        Computes the closest achievable resolution for the FCG sampling, given an image size
        and a chosen foveal radius.
        """
        max_x_prime = 0
        N_r = roh_max - roh_0  # max number of rings is half number of pixels minus half fovea
        distance_shrinking = True
        distance = np.inf
        while distance_shrinking:
            N_r -= 1
            a = np.exp((np.log(roh_max / roh_0) / N_r))
            max_delta_roh = (roh_0 + N_r) / np.floor(a ** (N_r - 1) * roh_0)
            fcx = fcy = np.floor((2 * (roh_0 + N_r) + 1) / 2)
            max_x_prime = int(np.floor(roh_max * max_delta_roh + fcx))
            max_y_prime = int(np.floor(roh_max * max_delta_roh + fcy))

            new_distance = np.abs(max_x_prime - desired_resolution)

            distance_shrinking = new_distance < distance

            distance = new_distance

        return N_r, max_x_prime

    def forward(self, img_x, fixations):
        transformed_image = self.warp_images(img_x, fixations,
                                             self.retina_coordinates,
                                             jitter_amount=self.jitter_amount,
                                             jitter_type=self.jitter_type)
        if len(fixations.shape) > 2:
            channel = transformed_image.shape[1]
            if channel == 1:
                scaled_image = torch.empty((img_x.shape[0], fixations.shape[1], self.img_target_size, self.img_target_size))
            else:
                scaled_image = torch.empty((img_x.shape[0], fixations.shape[1], channel, self.img_target_size, self.img_target_size))
            for i in range(fixations.shape[1]):
                if channel == 1:
                    image_batch = transformed_image[:, 0, i, :, :]
                    scaled_image[:, i, :, :] = torchvision.transforms.functional.resize(image_batch,
                                                                        (self.img_target_size, self.img_target_size), antialias=None)
                else:
                    image_batch = transformed_image[:, :, i, :, :]
                    scaled_image[:, i, :, :] = torchvision.transforms.functional.resize(image_batch,
                                                                        (self.img_target_size, self.img_target_size), antialias=None)
        else:
            scaled_image = torchvision.transforms.functional.resize(transformed_image,
                                                                        (self.img_target_size, self.img_target_size), antialias=None)
        if channel == 1:
            scaled_image = scaled_image.reshape(img_x.shape[0], fixations.shape[1], self.img_target_size * self.img_target_size)
        else:
            scaled_image = scaled_image.reshape(img_x.shape[0], fixations.shape[1], channel, self.img_target_size, self.img_target_size)
        return scaled_image