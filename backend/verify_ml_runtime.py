"""TensorFlow and ML Dependency Runtime Verification Script.

Systematically verifies:
A. Python runtime version (must be 3.11.x) and environment path.
B. TensorFlow import, tensor creation, and matrix multiplication.
C. Keras import, version alignment, and model-loading API presence.
D. NumPy operations and version compatibility.
E. Pillow RGB image creation, dimensional consistency, and NumPy conversion.
F. Physical device detection (CPU and optional GPU detection).
G. DenseNet deterministic preprocessing verification against ImageNet torch-mode normalization.
"""

import sys
from pathlib import Path
import numpy as np

def print_separator(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")

def verify_runtime() -> bool:
    all_passed = True

    # --------------------------------------------------------------------------
    # A. Python Environment Verification
    # --------------------------------------------------------------------------
    print_separator("A. Python Environment")
    py_ver = sys.version_info
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    
    if py_ver.major == 3 and py_ver.minor == 11:
        print("[PASS] Python version is 3.11.x as required.")
    else:
        print(f"[FAIL] Expected Python 3.11.x, but got {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
        all_passed = False

    expected_venv_subpath = ".venv-ml"
    if expected_venv_subpath in sys.executable:
        print(f"[PASS] Environment path correctly references '{expected_venv_subpath}'.")
    else:
        print(f"[INFO] Running in environment: {sys.executable}")

    # --------------------------------------------------------------------------
    # B. NumPy Verification
    # --------------------------------------------------------------------------
    print_separator("B. NumPy Verification")
    try:
        import numpy as np
        print(f"NumPy Version: {np.__version__}")
        
        # Test basic array operations
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        dot_res = np.dot(arr1, arr2)
        print(f"NumPy Dot Product [1,2,3,4] . [5,6,7,8] = {dot_res}")
        assert dot_res == 70.0, "Dot product computation mismatch"
        print("[PASS] NumPy imported and numerical operations completed successfully.")
    except Exception as exc:
        print(f"[FAIL] NumPy verification failed: {exc}")
        all_passed = False

    # --------------------------------------------------------------------------
    # C. Pillow Image Processing Verification
    # --------------------------------------------------------------------------
    print_separator("C. Pillow Image Processing")
    try:
        from PIL import Image
        print(f"Pillow (PIL) Version: {Image.__version__ if hasattr(Image, '__version__') else 'Imported'}")
        
        # Create a synthetic 224x224 RGB image matching model input spec
        test_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        img_array = np.array(test_img)
        print(f"Created RGB Image: size={test_img.size}, mode={test_img.mode}")
        print(f"Converted to NumPy array: shape={img_array.shape}, dtype={img_array.dtype}")
        assert img_array.shape == (224, 224, 3), f"Unexpected shape {img_array.shape}"
        assert img_array.dtype == np.uint8, f"Unexpected dtype {img_array.dtype}"
        print("[PASS] Pillow image creation and NumPy conversion verified.")
    except Exception as exc:
        print(f"[FAIL] Pillow verification failed: {exc}")
        all_passed = False

    # --------------------------------------------------------------------------
    # D. TensorFlow Verification
    # --------------------------------------------------------------------------
    print_separator("D. TensorFlow Verification")
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        # 1. Simple tensor operation
        t1 = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        t2 = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
        add_res = tf.add(t1, t2)
        print(f"Tensor Addition Result:\n{add_res.numpy()}")
        
        # 2. Matrix multiplication (matmul)
        matmul_res = tf.matmul(t1, t2)
        print(f"Tensor MatMul Result:\n{matmul_res.numpy()}")
        expected_matmul = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        assert np.allclose(matmul_res.numpy(), expected_matmul), "MatMul mismatch"
        print("[PASS] TensorFlow tensor creation, addition, and matmul executed successfully.")
    except Exception as exc:
        print(f"[FAIL] TensorFlow verification failed: {exc}")
        all_passed = False

    # --------------------------------------------------------------------------
    # E. Keras & DenseNet Preprocessing Verification
    # --------------------------------------------------------------------------
    print_separator("E. Keras & DenseNet Preprocessing")
    try:
        import keras
        print(f"Keras Version: {keras.__version__}")
        
        # 1. Check model loading API availability
        assert hasattr(keras.models, "load_model"), "keras.models.load_model not found"
        print("Keras load_model API: Available (keras.models.load_model)")
        
        # 2. Rigorous deterministic DenseNet preprocessing verification
        from tensorflow.keras.applications.densenet import preprocess_input
        
        # Expected ImageNet normalization constants (torch mode):
        # x_scaled = x / 255.0
        # x_norm = (x_scaled - mean) / std
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Test Case 1: Deterministic RGB tensor with distinct channel values
        # Input shape: (1, 224, 224, 3), dtype: float32, range [0, 255]
        # Channels: R=200.0, G=150.0, B=100.0
        input_channels = np.array([200.0, 150.0, 100.0], dtype=np.float32)
        test_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        test_tensor[:, :, :, :] = input_channels

        print(f"Input Tensor Shape: {test_tensor.shape}")
        print(f"Input Tensor Dtype: {test_tensor.dtype}")
        print(f"Input Value Range: [{test_tensor.min():.1f}, {test_tensor.max():.1f}]")
        print(f"Input Channel Values (RGB): R={input_channels[0]}, G={input_channels[1]}, B={input_channels[2]}")

        # Compute analytical expected result
        expected_channels = ((input_channels / 255.0) - imagenet_mean) / imagenet_std
        print(f"Analytically Expected Output (RGB): R={expected_channels[0]:.6f}, G={expected_channels[1]:.6f}, B={expected_channels[2]:.6f}")

        # Execute actual preprocess_input
        actual_output = preprocess_input(test_tensor.copy())
        actual_channels = actual_output[0, 0, 0, :]
        print(f"Actual Preprocessed Output (RGB):   R={actual_channels[0]:.6f}, G={actual_channels[1]:.6f}, B={actual_channels[2]:.6f}")

        max_diff = np.max(np.abs(actual_channels - expected_channels))
        print(f"Maximum Absolute Difference: {max_diff:.2e}")
        assert np.allclose(actual_output[0, 0, 0], expected_channels, atol=1e-5), "DenseNet preprocessing mismatch on test channels"

        # Test Case 2: Boundary check at 0.0 (minimum) and 255.0 (maximum)
        zeros_tensor = np.zeros((1, 2, 2, 3), dtype=np.float32)
        actual_zeros = preprocess_input(zeros_tensor.copy())
        expected_zeros = (0.0 - imagenet_mean) / imagenet_std
        assert np.allclose(actual_zeros[0, 0, 0], expected_zeros, atol=1e-5), "DenseNet preprocessing mismatch at 0.0"

        ones_255_tensor = np.ones((1, 2, 2, 3), dtype=np.float32) * 255.0
        actual_255 = preprocess_input(ones_255_tensor.copy())
        expected_255 = (1.0 - imagenet_mean) / imagenet_std
        assert np.allclose(actual_255[0, 0, 0], expected_255, atol=1e-5), "DenseNet preprocessing mismatch at 255.0"

        print("[PASS] DenseNet ImageNet torch-mode normalization verified with mathematical precision.")
    except Exception as exc:
        print(f"[FAIL] Keras / DenseNet preprocessing verification failed: {exc}")
        all_passed = False

    # --------------------------------------------------------------------------
    # F. Hardware & Device Discovery
    # --------------------------------------------------------------------------
    print_separator("F. Hardware & Physical Device Discovery")
    try:
        import tensorflow as tf
        physical_cpus = tf.config.list_physical_devices("CPU")
        print(f"Available Physical CPUs: {len(physical_cpus)} ({physical_cpus})")
        assert len(physical_cpus) > 0, "No CPU physical device detected"
        
        physical_gpus = tf.config.list_physical_devices("GPU")
        print(f"Available Physical GPUs: {len(physical_gpus)} ({physical_gpus})")
        if physical_gpus:
            print(f"[INFO] GPU acceleration is active: {physical_gpus}")
        else:
            print("[INFO] Operating in standard CPU-only mode (expected on native Windows TF 2.11+).")
        print("[PASS] Physical device detection completed.")
    except Exception as exc:
        print(f"[FAIL] Device discovery failed: {exc}")
        all_passed = False

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print_separator("Verification Summary")
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    else:
        print("ONE OR MORE VERIFICATION CHECKS FAILED.")
    
    return all_passed

if __name__ == "__main__":
    success = verify_runtime()
    sys.exit(0 if success else 1)
