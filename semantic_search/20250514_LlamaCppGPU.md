# How to use GPU setting in Llama-cpp to load model on available physical GPU

## Make sure that cuda is installed, and linked properly to the virtual conda env

- 1. in the terminal, check with the command `nvidia-smi` -> look for `CUDA version`, in my case, 12.8

- 2. check that the conda virtual env (e.g my env nanme is `workspace_2`) is linked to the installed cuda:

  - 2.1 in the terminal, run command: `echo $PATH`

  => should see some results as: `.../anaconda3/envs/workspace_2:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/libnvvp:...`

## Install `llama-cpp-python` with `gpu` enabled options

- 3. reference:

  - [3.1] https://github.com/abetlen/llama-cpp-python/issues/1535#issuecomment-2175924951

    - According to reference [3.1], if only using `pip install llama-cpp-python` for installing the `llama-cpp`, then all of the **gpu** related options (such as `n_gpu_layers`, `main_gpu`, `offload_kqv`, etc) in the model initiation function `Llama()` are **NOT** able to be used at all.

    - Therefore, at the beginning, we must add options to install `llama-cpp-python` with `gpu` options.

  - [3.2] https://pypi.org/project/llama-cpp-python/ -> section **Supported Backends** -> **Cuda**: to install `llama-cpp-python` with `gpu`, we must run:

        pip3 install pytorch torchvision torchaudio pytorch-cuda=11.8

        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>

  - In my case,  <cuda-version> = cu128 (see the 1., of section **Make sure that cuda is intalled...**).

  ## Initiate model gguf with `llama-cpp` to run on `gpu`

        model_pre_downloaded_path = "path/to/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
        gen_model = Llama(model_path=model_pre_downloaded_path, n_gpu_layers=-1, main_gpu=0, offload_kqv = True, n_ctx= 2048*8)

