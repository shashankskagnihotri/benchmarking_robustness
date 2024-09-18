import unittest
import numpy as np
from PIL import Image
import torch
from dataset.scene_flow import SceneFlowFlyingThingsDataset
import warnings

warnings.simplefilter('ignore')


class TestSceneFlowFlyingThingsDataset(unittest.TestCase):
    def setUp(self):
        # Set the path to your dataset here
        self.dataset_dir = '/pfs/work7/workspace/scratch/ma_aansari-team_project_fss2024_de/dataset/FlyingThings3D/'
        self.dataset = SceneFlowFlyingThingsDataset(self.dataset_dir, split='TEST')
        print("setUp done")

    def test_image_loading(self):
        """Test if all images can be loaded correctly as numpy arrays."""
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            # Check if 'left' image array is loaded correctly
            self.assertIsInstance(data['left'], torch.Tensor, f"Failed at index {i} for left image")
            # Check if 'right' image array is loaded correctly
            self.assertIsInstance(data['right'], torch.Tensor, f"Failed at index {i} for right image")
            # Optional: Check for 'occ_mask' and 'disp' based on your needs
            self.assertIsInstance(data['occ_mask'], np.ndarray, f"Failed at index {i} for occlusion mask left")
            self.assertIsInstance(data['occ_mask_right'], np.ndarray, f"Failed at index {i} for occlusion mask right")
            # self.assertTrue(data['disp'].size > 0, f"Disparity map left at index {i} is empty")
            # self.assertTrue(data['disp_right'].size > 0, f"Disparity map right at index {i} is empty")

            # # Assert that images are not empty
            # self.assertGreater(data['left'].size, 0, f"Left image at index {i} is empty")
            # self.assertGreater(data['right'].size, 0, f"Right image at index {i} is empty")
            if i % 100 == 0:
                print(f"Tested index {i}/{len(self.dataset)} successfully.")

if __name__ == '__main__':
    unittest.main()
