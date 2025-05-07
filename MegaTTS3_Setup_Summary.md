# MegaTTS3 Setup Summary

This document summarizes the successful setup of the MegaTTS3 text-to-speech system on a Mac Studio M1 Max, leveraging Metal Performance Shaders (MPS) for GPU acceleration. It details the installation steps, which differ from the official documentation due to the need for MPS support, and covers configuration, performance, audio quality, management of temporary files, and recommendations for ongoing use.

## Installation Steps

The official MegaTTS3 documentation (e.g., [MegaTTS3 README](https://github.com/ByteDance/MegaTTS3)) likely assumes CUDA (NVIDIA GPU) or CPU environments, requiring modifications for the M1 Max’s MPS. Below are the steps taken to install and configure MegaTTS3 on macOS, addressing environment setup, MPS support, and specific issues like `~/.zshrc` permissions.

1. **Create Conda Environment**:
   - Created a Conda environment named `megatts3-env` with Python 3.10, as the official guide may not specify macOS-specific versions:
     ```bash
     conda create -n megatts3-env python=3.10
     conda activate megatts3-env
     ```

2. **Install PyTorch with MPS Support**:
   - Installed PyTorch 2.7.0, which supports MPS on M1 Max, unlike the official guide’s likely CUDA focus:
     ```bash
     pip install torch torchvision torchaudio
     ```
   - Verified MPS support:
     ```bash
     python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
     ```
     - Output: `2.7.0 True`

3. **Install Dependencies**:
   - Installed project dependencies, following the official `requirements.txt` but with macOS-specific adjustments:
     ```bash
     pip install -r requirements.txt
     conda install -y -c conda-forge pynini==2.1.5
     pip install WeTextProcessing==1.0.3
     conda install -c conda-forge ffmpeg
     ```
   - Added Gradio and Pydantic for the web interface:
     ```bash
     pip install pydantic==2.5.0 gradio==5.23.3
     ```
     or 如果你在中国？前面包括后面的安装可以参考，在pip install {命令最后面指定清华大学} -i https://pypi.tuna.tsinghua.edu.cn/simple
     ```bash
     pip install pydantic==2.5.0 gradio==5.23.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
     ```

4. **Download Checkpoints**:
   - Downloaded MegaTTS3 checkpoints, as per the official guide:
     ```bash
     huggingface-cli download ByteDance/MegaTTS3 --local-dir ./checkpoints --local-dir-use-symlinks False
     ```
   - Verified files exist:
     ```bash
     ls -l checkpoints
     ```

5. **Set `PYTHONPATH`**:
   - Encountered a `permission denied: /Users/mio/.zshrc` error when setting `PYTHONPATH`. Fixed permissions:
     ```bash
     sudo chown mio:staff ~/.zshrc
     chmod 644 ~/.zshrc
     ```
   - Added `PYTHONPATH` to include the project directory:
     ```bash
     echo 'export PYTHONPATH="/Users/mio/MegaTTS3:$PYTHONPATH"' >> ~/.zshrc
     source ~/.zshrc
     ```
   - Used a temporary workaround during setup:
     ```bash
     export PYTHONPATH="/Users/mio/MegaTTS3:$PYTHONPATH"
     ```

6. **Modify Scripts for MPS Support**:
   - **Edited `tts/gradio_api.py`**:
     - Updated `model_worker` to use MPS:
       ```python
       device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
       print(f"Worker running on device: {device}")
       infer_pipe = MegaTTS3DiTInfer(device=device)
       ```
     - Modified the main block for Gradio:
       ```python
       devices = [0] if torch.backends.mps.is_available() else None
       ```
   - **Edited `tts/infer_cli.py`**:
     - Changed device initialization to prioritize MPS:
       ```python
       if device is None:
           device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
       else:
           device = torch.device(device)
       self.device = device
       ```
     - Added debug print in `build_model`:
       ```python
       print(f"Models loaded on device: {device}")
       ```
     - Fixed syntax errors:
       - Corrected `hp_dur_ model` to `hp_dur_model` in `build_model`.
       - Corrected `np pad` to `np.pad` in `preprocess`.

7. **Resolve Runtime Issues**:
   - Fixed `ModuleNotFoundError: No module named 'torch'` by ensuring `megatts3-env` was active:
     ```bash
     conda activate megatts3-env
     ```
   - Addressed syntax errors in `infer_cli.py` through iterative debugging.

8. **Tune Parameters for Quality**:
   - Initial runs with `time python tts/gradio_api.py` were faster but produced incorrect voice characteristics.
   - Adjusted Gradio parameters to match quality of `python tts/gradio_api.py`:
     - `Infer Timestep`: 32
     - `Intelligibility Weight`: 1.4
     - `Similarity Weight`: 2
   - Verified performance (2x faster with MPS) and quality via the Gradio UI (`http://0.0.0.0:7929`).

**Differences from Official Documentation**:
- **MPS Support**: Official guide likely targets CUDA or CPU, requiring manual changes to `gradio_api.py` and `infer_cli.py` for MPS.
- **Environment Setup**: Specified Python 3.10 and PyTorch 2.7.0 for macOS compatibility.
- **Permission Fixes**: Handled `~/.zshrc` permission issues not mentioned in the official guide.
- **Parameter Tuning**: Required specific Gradio parameters to achieve optimal audio quality, not detailed in the official setup.
- **Debugging**: Fixed syntax errors in `infer_cli.py` not present in the official codebase.

## Overview

MegaTTS3 is now fully operational with optimal performance and audio quality on the Mac Studio M1 Max. The setup leverages the M1 Max GPU via MPS, achieving a ~2x speed-up compared to CPU-based runs, with high-quality audio output matching the reference voice.

## Configuration

- **Scripts**:
  - `tts/gradio_api.py`: Runs the Gradio web interface.
  - `tts/infer_cli.py`: Core inference logic, modified for MPS.
  - Both use `device="mps"` for GPU acceleration.

- **Gradio Interface Parameters**:
  - `Infer Timestep`: 50
  - `Intelligibility Weight`: 1.0
  - `Similarity Weight`: 5.0

- **Environment**:
  - Conda environment: `megatts3-env` (Python 3.10).
  - PyTorch: 2.7.0 with MPS support.
  - `PYTHONPATH`: `/Users/mio/MegaTTS3`.

- **Command to Run**:
  ```bash
  conda activate megatts3-env
  export PYTHONPATH="/Users/mio/MegaTTS3:$PYTHONPATH"
  python tts/gradio_api.py
  ```
  - Access the Gradio UI at `http://0.0.0.0:7929`.
  - Upload a `.wav` file (24kHz), corresponding `.npy` (WaveVAE latent), and input text (e.g., "Hello, this is a test.").

## Performance and Quality

- **Performance**:
  - `time python tts/gradio_api.py` is ~2x faster than `python tts/gradio_api.py` due to MPS.
  - M1 Max GPU accelerates inference significantly.

- **Audio Quality**:
  - Initial `time python tts/gradio_api.py` runs had poor voice characteristics.
  - Resolved with `Infer Timestep=50`, `Intelligibility Weight=1.0`, `Similarity Weight=5.0`.

- **Verification**:
  - Terminal output: `Models loaded on device: mps`, `Worker running on device: mps`.
  - Activity Monitor confirms GPU activity.

## Managing Temporary Files

MegaTTS3 generates minimal temporary files:

- **Generated Files**:
  - **Converted `.wav` Files**:
    - Non-`.wav` inputs (e.g., `input.mp3`) create `.wav` files (e.g., `input.wav`).
    - These persist and may be "garbage" if unneeded.
  - **Modified `.wav` Files**:
    - `cut_wav` truncates `.wav` files to 28 seconds, overwriting them.

- **Gradio Cache**:
  - Gradio stores uploads and outputs in `~/.gradio/`.
  - Cleaned up on clean shutdown (`Ctrl+C`), but may persist if crashed.

- **Mitigation Strategies**:
  - **Use `.wav` Inputs**: Upload 24kHz `.wav` files to avoid conversion.
  - **Manual Cleanup**:
    ```bash
    rm *.wav  # Caution: Only delete unneeded .wav files
    rm -rf ~/.gradio/*  # Clear Gradio cache
    ```
  - **Modify `convert_to_wav`** (optional):
    - Edit `tts/infer_cli.py`:
      ```python
      import tempfile
      def convert_to_wav(wav_path):
          if not os.path.exists(wav_path):
              print(f"The file '{wav_path}' does not exist.")
              return
          if not wav_path.endswith(".wav"):
              out_path = os.path.join(tempfile.gettempdir(), os.path.basename(os.path.splitext(wav_path)[0]) + ".wav")
              audio = AudioSegment.from_file(wav_path)
              audio.export(out_path, format="wav")
              print(f"Converted '{wav_path}' to '{out_path}'")
              return out_path
          return wav_path
      ```
    - Update `model_worker` in `tts/gradio_api.py`:
      ```python
      wav_path = convert_to_wav(inp_audio_path)
      if wav_path:
          cut_wav(wav_path, max_len=28)
      ```
    - Uses `/tmp`, cleaned by macOS periodically.

- **Check Files**:
  - Monitor directory:
    ```bash
    ls -l .
    ```
  - Verify `.wav` integrity:
    ```bash
    ffprobe your_input.wav
    ```

## Recommendations

1. **Persist Environment Settings**:
   ```bash
   echo 'export PYTHONPATH="/Users/mio/MegaTTS3:$PYTHONPATH"' >> ~/.zshrc
   echo 'conda activate megatts3-env' >> ~/.zshrc
   source ~/.zshrc
   ```

2. **Automate Execution and Cleanup**:
   - Create `run_megatts3.sh`:
     ```bash
     #!/bin/bash
     conda activate megatts3-env
     export PYTHONPATH="/Users/mio/MegaTTS3:$PYTHONPATH"
     python tts/gradio_api.py
     rm -f *.wav  # Caution: Only run if unneeded .wav files
     ```
   - Make executable:
     ```bash
     chmod +x run_megatts3.sh
     ```

3. **Optimize Parameters**:
   - Test `Infer Timestep=32` or `Similarity Weight=3.0` for better quality.
   - Example: `Infer Timestep=32`, `Intelligibility Weight=1.4`, `Similarity Weight=3.0`.

4. **Backup Environment**:
   ```bash
   conda env export > megatts3-env.yml
   ```

5. **Monitor Resources**:
   - Use Activity Monitor for GPU usage.
   - Verify MPS:
     ```bash
     python -c "import torch; print(torch.backends.mps.is_available())"
     ```

## Conclusion

The MegaTTS3 setup on the Mac Studio M1 Max is fully operational, achieving fast inference (~2x speed-up with MPS) and high-quality audio. The installation required MPS-specific modifications, permission fixes, and parameter tuning, diverging from the official CUDA-focused guide. Temporary files are minimal and manageable. For issues, check the [MegaTTS3 README](https://github.com/ByteDance/MegaTTS3) or contact maintainers.
