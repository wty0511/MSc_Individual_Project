
#!/bin/bash

# 设置Git用户名和邮箱
git config --global user.name wty0511
git config --global user.email 15652198208@163.com

# 检查是否已存在SSH密钥
if [ ! -f ~/.ssh/id_rsa.pub ]; then
    # 如果不存在，生成新的SSH密钥
    ssh-keygen -t rsa -b 4096 -C "YOUR_GIT_EMAIL" -f ~/.ssh/id_rsa -N ""
    echo "新的SSH密钥已经生成。"
else
    echo "已经存在SSH密钥，无需生成。"
fi

# 打印出新的或现有的公钥，这样你就可以将其添加到你的Git服务器
echo "你的公钥如下："
cat ~/.ssh/id_rsa.pub



# 检查是否已经安装Anaconda
if ! command -v conda &> /dev/null; then
    # 如果没有安装Anaconda，下载并安装
    echo "Anaconda未安装，现在开始安装..."
    cd ~
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
    bash Anaconda3-2022.10-Linux-x86_64.sh
    # 重启shell以更新环境变量
    export PATH=~/anaconda3/bin:$PATH
    source ~/.bashrc
else
    echo "Anaconda已经安装，无需重新安装。"
fi

# 创建新的Anaconda环境，名为myenv



# 检查是否已经安装了名为myenv的环境
if ! conda env list | grep -q 'pyenv'; then
    # 如果没有，创建它
    echo "环境'myenv'不存在，现在开始创建..."
    conda create --name pyenv
else
    echo "环境'myenv'已经存在。"
fi

# 定义要安装的包和版本
packages=(antlr4-python3-runtime==4.8 appdirs==1.4.4 audioread==3.0.0 brotlipy==0.7.0 certifi==2022.12.7 cffi==1.15.1 chardet==5.1.0 charset-normalizer==3.1.0 click==8.1.3 colorlog==6.7.0 contourpy==1.0.7 cycler==0.11.0 decorator==5.1.1 docker-pycreds==0.4.0 dpcpp-cpp-rt==2023.0.0 filelock==3.12.0 fonttools==4.39.3 future==0.18.3 gitdb==4.0.10 GitPython==3.1.31 h5py==3.8.0 huggingface-hub==0.13.4 hydra-colorlog==1.2.0 hydra-core==1.1.0 idna==3.4 importlib-resources==5.12.0 intel-cmplr-lib-rt==2023.0.0 intel-cmplr-lic-rt==2023.0.0 intel-opencl-rt==2023.0.0 intel-openmp==2023.0.0 joblib==1.2.0 kiwisolver==1.4.4 librosa==0.8.1 llvmlite==0.39.1 markdown-it-py==2.2.0 matplotlib==3.7.1 mdurl==0.1.2 mir-eval==0.6 mkl==2023.1.0 mkl-fft==1.3.1 mkl-service==2.4.0 numba==0.56.4 numpy==1.22.4 nvidia-cublas-cu11==11.10.3.66 nvidia-cuda-nvrtc-cu11==11.7.99 nvidia-cuda-runtime-cu11==11.7.99 nvidia-cudnn-cu11==8.5.0.96 omegaconf==2.1.2 packaging==23.1 pandas==2.0.1 pathtools==0.1.2 Pillow==9.4.0 platformdirs==3.5.0 pooch==1.7.0 protobuf==4.22.3 psutil==5.9.5 pycparser==2.21 Pygments==2.15.1 pyparsing==3.0.9 python-dateutil==2.8.2 pytz==2023.3 PyYAML==5.3.1 regex==2023.3.23 requests==2.29.0 resampy==0.4.2 rich==13.3.4 scikit-learn==1.2.2 scipy==1.10.1 sdeint==0.3.0 seaborn==0.12.2 sentry-sdk==1.21.1 setproctitle==1.3.2 six==1.16.0 smmap==5.0.0 soundfile==0.12.1 threadpoolctl==3.1.0 tokenizers==0.13.3 torch==1.13.1 torchaudio==0.13.1 torchvision==0.14.1 tqdm==4.65.0 transformers==4.28.1 typing_extensions==4.5.0 tzdata==2023.3 urllib3==1.26.15 wandb==0.15.0 zipp==3.15.0)


# 激活环境
source activate pyenv


# 遍历数组，安装每个包
for package in "${packages[@]}"; do
    # 检查包是否已经被安装
    if ! pip show "${package%%==*}" > /dev/null; then
        echo "正在安装包：$package"
        pip install "$package"
    else
        echo "包已经安装：$package"
    fi
done
