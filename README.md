# Bitnet.cpp
## Introduction
Bitnet.cpp is a specialized inference framework designed for Bitnet ternary models, optimized for efficient CPU-based inference. Bitnet.cpp provides an end-to-end inference solution by integrating specialized compute kernels with the llama.cpp framework. Bitnet.cpp supports multiple quantization methods and various model sizes, providing low memory usage and low power consumption without compromising on performance or accuracy.

## Installation
### Requirements
- conda
- cmake>=3.22

### Build from source
1. Clone the repo
```bash
git clone https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Create a new conda environment(Recommended)
```bash
conda env create -f environment.yaml
conda activate bitnet-cpp
```
3. Build the project
```bash
# Download the model from Hugging Face, convert it to quantized gguf format, and build the project
python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large -q tl1
# Or you can manually download the model and run with local path
huggingface-cli download 1bitLLM/bitnet_b1_58-large --local-dir models/bitnet_b1_58-large 
python setup_env.py -md models/bitnet_b1_58-large -q tl1
```
> usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR]
>                     [--quant-type {i2_s,tl1}] [--quant-embd]
> optional arguments:
>   -h, --help            show this help message and exit
>   --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B}
>                         Model used for inference
>
>   --model-dir MODEL_DIR, -md MODEL_DIR
>                         Directory to save/load the model
>
>   --log-dir LOG_DIR, -ld LOG_DIR
>                         Directory to save the logging info
>   --quant-type {i2_s,tl1}, -q {i2_s,tl1}
>                         Quantization type
>   --quant-embd          Quantize the embeddings to f16

## Usage
### Basic usage
```bash
# Run inference with a given prompt
python run_inference.py -m models/bitnet_b1_58-large/ggml-model-i2_s.gguf -p "Microsoft Corporation is"
```
> Example output:  
> Microsoft Corporation is an American software company headquartered in Redmond, Washington, United States. The company is a subsidiary of Microsoft Corporation. Microsoft is an American software company that designs and develops computer software and services. The company was founded in 1975 by Bill Gates. Microsoft has its headquarters in Redmond, Washington, United States.
Microsoft Corporation's business is mainly focused on software development and application development. The company is the largest software company in the world and a member of the Microsoft Group. The company was founded by Bill Gates, Paul Allen, and Steve Ballmer.

<h3 align="center">
    <img src="media/demo.gif">
</h3>



### Advanced usage
// TODO
We provide a series of tools that allow you to manually tune the kernel for your own device.

We also provide scipts to generate fake bitnet models with different sizes, making it easier to test the performance of the framework on your machine.

```bash
python utils/generate-fake-bitnet-model.py models/bitnet_b1_58-large --outfile models/fake-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M
```