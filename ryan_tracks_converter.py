import io
from typing import Union, List, Dict, Optional
import torch
import numpy as np
from wan.utils.motion import *
def save_weird_tracks_file(tracks_tn3, path):
    """
    Convert TN3 torch tensor to NT13 format and save as .pth file.
    
    Args:
        tracks_tn3: torch tensor of shape (T, N, 3) where T=frames, N=points, 3=TXY
        path: output path for .pth file
    """
    import torch
    import numpy as np
    import io
    
    # Convert TN3 to NT13: transpose first two dims, add empty dim at index 2
    tracks_nt3 = tracks_tn3.transpose(0, 1)  # (T, N, 3) -> (N, T, 3)
    tracks_nt13 = tracks_nt3.unsqueeze(2)    # (N, T, 3) -> (N, T, 1, 3)
    
    # Convert to numpy and save as NPZ in memory
    tracks_np = tracks_nt13.cpu().numpy()
    bytes_io = io.BytesIO()
    np.savez_compressed(bytes_io, array=tracks_np)
    
    # Save the bytes data as .pth file
    torch.save(bytes_io.getvalue(), path)

def test_save_weird_tracks_file():
    """Test that save_weird_tracks_file correctly converts TN3 to NT13 format"""
    import torch
    import numpy as np
    import tempfile
    import os
    
    # Create test data: TN3 tensor (5 frames, 10 points, 3 coords)
    T, N = 5, 10
    tracks_tn3 = torch.randn(T, N, 3)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        # Save using our function
        save_weird_tracks_file(tracks_tn3, temp_path)
        
        # Load back using the existing loading mechanism
        loaded_data = torch.load(temp_path)
        tracks_np = unzip_to_array(loaded_data)
        
        # Verify shape is NT13
        expected_shape = (N, T, 1, 3)
        assert tracks_np.shape == expected_shape, f"Expected shape {expected_shape}, got {tracks_np.shape}"
        
        # Verify data matches (convert back to TN3 for comparison)
        loaded_tn3 = torch.from_numpy(tracks_np).squeeze(2).transpose(0, 1)  # NT13 -> TN3
        
        # Check if tensors are close (accounting for float precision)
        assert torch.allclose(tracks_tn3, loaded_tn3, atol=1e-6), "Data doesn't match after save/load"
        
        print(f"✓ Test passed!")
        print(f"  Original shape: {tracks_tn3.shape}")
        print(f"  Saved shape: {tracks_np.shape}")
        print(f"  Data integrity: OK")
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
