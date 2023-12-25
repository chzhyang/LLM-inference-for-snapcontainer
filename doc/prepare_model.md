# Prepare Model

| Source       | Model | Download |
| ----------- | --- | -- |
| Hugging Face | [Meta/Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [THUDM/chatglm2-6b](https://huggingface.co/meta-llama/chatglm2-6b) | [Method](#download-hugging-face-model) |
|  |  |  |

## Download Hugging Face model

### 1. Install git-lfs

[git lfs](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing), for ubuntu:

```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git lfs --version
```

### 2. Download model

Set model repository path

```shell
export MODEL_REPOSITORY="~/models"
```

Download model to MODEL_REPOSITORY

[THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b), FP16

```shell
git lfs clone https://huggingface.co/THUDM/chatglm2-6b $MODEL_REPOSITORY/chatglm2-6b
```

[Meta/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), FP16

```shell
git lfs clone https://huggingface.co/meta-llama/Llama-2-7b-hf $MODEL_REPOSITORY/Llama-2-7b-hf
```