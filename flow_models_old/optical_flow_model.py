class OpticalFlowModel:
    def __init__(self):
        self.cache = {}

    def compute_optical_flow(self, image1, image2):
        """
        Args:
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
        Returns:
            flow: (B, 2, H, W)
        """
        

    def forward(self, image1, image2, idx1=None, idx2=None):
        """
        Args:
            image1: (B, 3, H, W) [0.0, 1.0]
            image2: (B, 3, H, W) [0.0, 1.0]
            idx1: int
            idx2: int
        """

        if idx1 is not None and idx2 is not None:
            key = (idx1, idx2)
            if key in self.cache:
                return self.cache[key]
    
        # Compute optical flow between image1 and image2
        flow = self.compute_optical_flow(image1, image2)

        if idx1 is not None and idx2 is not None:
            self.cache[(idx1, idx2)] = flow

        return flow