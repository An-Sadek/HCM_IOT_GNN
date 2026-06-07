import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DynamicTrafficDataset(Dataset):
    def __init__(self, velocity_path, los_path, window_size=12, horizon=1):
        """
        window_size: Số lượng time steps quá khứ dùng làm đầu vào (Ví dụ: 12 bước = 6 tiếng)
        horizon: Số lượng time steps tương lai cần dự đoán
        """
        # Load bằng mmap_mode='r' giúp không tốn tí RAM nào ở bước này
        self.velocity_data = np.load(velocity_path, mmap_mode='r')
        self.los_data = np.load(los_path, mmap_mode='r')
        
        self.window_size = window_size
        self.horizon = horizon
        
        # Tổng số time step có sẵn trong data
        self.total_timesteps = self.velocity_data.shape[0]
        
        # Tính toán số lượng sample thực tế có thể tạo ra từ cơ chế sliding window
        self.num_samples = self.total_timesteps - self.window_size - self.horizon + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Xác định khoảng index cho input (X) và target (y)
        start_idx = idx
        end_idx = idx + self.window_size
        target_idx = end_idx + self.horizon - 1
        
        # Đọc dữ liệu từ đĩa thông qua memory-map
        # Giả sử bạn muốn ghép cả velocity và LOS lại làm features đầu vào
        x_vel = self.velocity_data[start_idx:end_idx]  # Shape: (window_size, num_segments, 2)
        x_los = self.los_data[start_idx:end_idx]       # Shape: (window_size, num_segments, 2)
        
        # Tạo feature X bằng cách concat chúng lại theo trục channel cuối cùng
        # Shape kết quả: (window_size, num_segments, 4)
        X = np.concatenate([x_vel, x_los], axis=-1)
        
        # Lấy target (ví dụ: lấy giá trị LOS thực tế ở tương lai làm nhãn)
        # Nhãn chỉ cần kênh dữ liệu gốc, bỏ mask (kênh 0)
        y = self.los_data[target_idx, :, 0] # Shape: (num_segments,)
        
        # Chuyển sang PyTorch Tensor
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)