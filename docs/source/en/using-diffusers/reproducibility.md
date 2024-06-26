<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Create reproducible pipelines

[[open-in-colab]]

Reproducibility is important for testing, replicating results, and can even be used to [improve image quality](reusing_seeds). However, the randomness in diffusion models is a desired property because it allows the pipeline to generate different images every time it is run. While you can't expect to get the exact same results across platforms, you can expect results to be reproducible across releases and platforms within a certain tolerance range. Even then, tolerance varies depending on the diffusion pipeline and checkpoint.

This is why it's important to understand how to control sources of randomness in diffusion models or use deterministic algorithms.

<Tip>

💡 We strongly recommend reading PyTorch's [statement about reproducibility](https://pytorch.org/docs/stable/notes/randomness.html):

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.

</Tip>

## Control randomness

During inference, pipelines rely heavily on random sampling operations which include creating the
Gaussian noise tensors to denoise and adding noise to the scheduling step.

Take a look at the tensor values in the [`DDIMPipeline`] after two inference steps:

```python
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np").images
print(np.abs(image).sum())
```

Running the code above prints one value, but if you run it again you get a different value. What is going on here?

Every time the pipeline is run, [`torch.randn`](https://pytorch.org/docs/stable/generated/torch.randn.html) uses a different random seed to create Gaussian noise which is denoised stepwise. This leads to a different result each time it is run, which is great for diffusion pipelines since it generates a different random image each time.

But if you need to reliably generate the same image, that'll depend on whether you're running the pipeline on a CPU or GPU.

### CPU

To generate reproducible results on a CPU, you'll need to use a PyTorch [`Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) and set a seed:

```python
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)

# create a generator for reproducibility
generator = torch.Generator(device="cpu").manual_seed(0)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

Now when you run the code above, it always prints a value of `1491.1711` no matter what because the `Generator` object with the seed is passed to all the random functions of the pipeline.

If you run this code example on your specific hardware and PyTorch version, you should get a similar, if not the same, result.

<Tip>

💡 It might be a bit unintuitive at first to pass `Generator` objects to the pipeline instead of
just integer values representing the seed, but this is the recommended design when dealing with
probabilistic models in PyTorch, as `Generator`s are *random states* that can be
passed to multiple pipelines in a sequence.

</Tip>

### GPU

Writing a reproducible pipeline on a GPU is a bit trickier, and full reproducibility across different hardware is not guaranteed because matrix multiplication - which diffusion pipelines require a lot of - is less deterministic on a GPU than a CPU. For example, if you run the same code example above on a GPU:

```python
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)
ddim.to("cuda")

# create a generator for reproducibility
generator = torch.Generator(device="cuda").manual_seed(0)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

The result is not the same even though you're using an identical seed because the GPU uses a different random number generator than the CPU.

To circumvent this problem, 🧨 Diffusers has a [`~diffusers.utils.torch_utils.randn_tensor`] function for creating random noise on the CPU, and then moving the tensor to a GPU if necessary. The `randn_tensor` function is used everywhere inside the pipeline, allowing the user to **always** pass a CPU `Generator` even if the pipeline is run on a GPU.

You'll see the results are much closer now!

```python
import torch
from diffusers import DDIMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddim = DDIMPipeline.from_pretrained(model_id, use_safetensors=True)
ddim.to("cuda")

# create a generator for reproducibility; notice you don't place it on the GPU!
generator = torch.manual_seed(0)

# run pipeline for just two steps and return numpy tensor
image = ddim(num_inference_steps=2, output_type="np", generator=generator).images
print(np.abs(image).sum())
```

<Tip>

💡 If reproducibility is important, we recommend always passing a CPU generator.
The performance loss is often neglectable, and you'll generate much more similar
values than if the pipeline had been run on a GPU.

</Tip>

Finally, for more complex pipelines such as [`UnCLIPPipeline`], these are often extremely
susceptible to precision error propagation. Don't expect similar results across
different GPU hardware or PyTorch versions. In this case, you'll need to run
exactly the same hardware and PyTorch version for full reproducibility.

## Deterministic algorithms

You can also configure PyTorch to use deterministic algorithms to create a reproducible pipeline. However, you should be aware that deterministic algorithms may be slower than nondeterministic ones and you may observe a decrease in performance. But if reproducibility is important to you, then this is the way to go!

Nondeterministic behavior occurs when operations are launched in more than one CUDA stream. To avoid this, set the environment variable [`CUBLAS_WORKSPACE_CONFIG`](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) to `:16:8` to only use one buffer size during runtime.

PyTorch typically benchmarks multiple algorithms to select the fastest one, but if you want reproducibility, you should disable this feature because the benchmark may select different algorithms each time. Lastly, pass `True` to [`torch.use_deterministic_algorithms`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) to enable deterministic algorithms.

```py
import os
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

Now when you run the same pipeline twice, you'll get identical results.

```py
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
g = torch.Generator(device="cuda")

prompt = "A bear is playing a guitar on Times Square"

g.manual_seed(0)
result1 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

g.manual_seed(0)
result2 = pipe(prompt=prompt, num_inference_steps=50, generator=g, output_type="latent").images

print("L_inf dist =", abs(result1 - result2).max())
"L_inf dist = tensor(0., device='cuda:0')"
```
