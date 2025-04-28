# MegaTTS3 Setup Summary

This document summarizes the successful setup of the MegaTTS3 text-to-speech system on a Mac Studio M1 Max, leveraging Metal Performance Shaders (MPS) for GPU acceleration. It includes details on configuration, performance, audio quality, management of temporary files, and recommendations for ongoing use.

## Overview

MegaTTS3 is now fully operational with optimal performance and audio quality, achieved through specific parameter tuning and environment configuration. The setup uses the `megatts3-env` Conda environment, with MPS-enabled scripts for fast inference on the M1 Max GPU.

## Configuration

- **Scripts**:
  - `tts/gradio_api.py`: Runs the Gradio web interface for text-to-speech inference.
  - `tts/infer_cli.py`: Core inference logic for MegaTTS3, modified to use MPS.
  - Both scripts are configured to use `device="mps"` for GPU acceleration.

- **Gradio Interface Parameters**:
  - `Infer Timestep`: 50 (controls diffusion steps for quality).
  - `Intelligibility Weight`: 1.0 (balances phoneme clarity).
  - `Similarity Weight`: 5.0 (ensures voice matches reference audio).

- **Environment**:
  - Conda environment: `megatts3-env` (Python 3.10).
  - PyTorch: 2.7.0 with MPS support (`torch.backends.mps.is_available() == True`).
  - `PYTHONPATH`: `/Users/mio/MegaTTS3` to include project modules.

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
  - Running `time python tts/gradio_api.py` is approximately 2x faster than `python tts/gradio_api.py` due to efficient MPS utilization.
  - The M1 Max GPU accelerates inference significantly compared to CPU.

- **Audio Quality**:
  - Initial runs with `time python tts/gradio_api.py` produced incorrect voice characteristics.
  - Resolved by tuning parameters to `Infer Timestep=50`, `Intelligibility Weight=1.0`, `Similarity Weight=5.0`, matching the quality of `python tts/gradio_api.py`.

- **Verification**:
  - Terminal output confirms MPS usage: `Models loaded on device: mps` and `Worker running on device: mps`.
  - Activity Monitor shows GPU activity during inference.

## Managing Temporary Files

MegaTTS3 generates minimal temporary files, but some may persist:

- **Generated Files**:
  - **Converted `.wav` Files**:
    - If a non-`.wav` input (e.g., `input.mp3`) is uploaded, `convert_to_wav` creates a `.wav` file (e.g., `input.wav`) in the same directory.
    - These files persist and may be considered "garbage" if not needed.
  - **Modified `.wav` Files**:
    - `cut_wav` truncates `.wav` files to 28 seconds, overwriting the original.
    - No additional files are created here.

- **Gradio Cache**:
  - Gradio stores uploaded files (`.wav`, `.npy`) and outputs in a temporary cache (e.g., `~/.gradio/`).
  - These are typically cleaned up on clean shutdown (`Ctrl+C`), but may remain if the process crashes.

- **Mitigation Strategies**:
  - **Use `.wav` Inputs**: Upload 24kHz `.wav` files to avoid `convert_to_wav` creating new files.
  - **Manual Cleanup**:
    ```bash
    rm *.wav  # Caution: Only delete unneeded .wav files
    rm -rf ~/.gradio/*  # Clear Gradio cache
    ```
  - **Modify `convert_to_wav`** (optional):
    - Edit `tts/infer_cli.py` to store converted `.wav` files in a temporary directory:
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
    - This uses `/tmp`, which macOS cleans periodically.

- **Check Files**:
  - Monitor the working directory:
    ```bash
    ls -l .
    ```
  - Verify input `.wav` integrity:
    ```bash
    ffprobe your_input.wav
    ```

## Recommendations

1. **Persist Environment Settings**:
   - Add to `~/.zshrc` for convenience:
     ```bash
     echo 'export PYTHONPATH="/Users/mio/MegaTTS3:$PYTHONPATH"' >> ~/.zshrc
     echo 'conda activate megatts3-env' >> ~/.zshrc
     source ~/.zshrc
     ```

2. **Automate Execution and Cleanup**:
   - Create a script (`run_megatts3.sh`):
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
   - Test higher `Infer Timestep` (e.g., 75) or `Similarity Weight` (e.g., 6.0) for improved quality, balancing with performance.
   - Example: `Infer Timestep=75`, `Intelligibility Weight=1.0`, `Similarity Weight=6.0`.

4. **Backup Environment**:
   - Save Conda environment:
     ```bash
     conda env export > megatts3-env.yml
     ```

5. **Monitor Resources**:
   - Use Activity Monitor to confirm GPU usage.
   - Verify MPS support:
     ```bash
     python -c "import torch; print(torch.backends.mps.is_available())"
     ```

## Conclusion

The MegaTTS3 setup is fully operational with fast inference (2x speed-up using MPS) and high-quality audio output. Temporary files are minimal and manageable with proper input choices and cleanup strategies. This configuration leverages the M1 Max GPU effectively, providing a robust platform for text-to-speech tasks.

For issues (e.g., file accumulation, quality changes), contact the project maintainers or check the [MegaTTS3 README](https://github.com/ByteDance/MegaTTS3) for additional guidance.