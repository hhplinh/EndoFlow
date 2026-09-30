class DepthModel:
    def __init__(self, *args, **kwargs):
        self.cache = {}

    def compute_depth(self, image_left, image_right, fx, baseline):
        """
        Args:
            image_left: (3, H, W) [0.0, 1.0]
            image_right: (3, H, W) [0.0, 1.0]
            fx: float
            baseline: float
        Returns:
            depth: (H, W)
        """

        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, image_left, image_right, fx, baseline, idx=None):
        """
        Args:
            image_left: (3, H, W) [0.0, 1.0]
            image_right: (3, H, W) [0.0, 1.0]
            fx: float
            baseline: float
            idx: int
        Returns:
            depth: (H, W)
        """

        if idx is not None and idx in self.cache:
                return self.cache[idx]

        # Compute depth from stereo images
        depth = self.compute_depth(image_left, image_right, fx, baseline)

        if idx is not None:
            self.cache[idx] = depth

        return depth

    def forward_w_camera(self, camera, idx=None):
        """
        Args:
            camera: Camera
            idx: int
        Returns:
            depth: (H, W)
        """

        if idx is not None and idx in self.cache:
                return self.cache[idx]

        # Compute depth from stereo images
        depth = self.compute_depth(camera.original_image, camera.original_image_right, camera.view_image_transform[0, 0], camera.baseline)

        if idx is not None:
            self.cache[idx] = depth

        return depth