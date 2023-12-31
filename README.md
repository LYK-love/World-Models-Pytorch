# Pytorch implementation of the "WorldModels"

Paper: Ha and Schmidhuber, "World Models", 2018. https://doi.org/10.5281/zenodo.1207631. For a quick summary of the paper and some additional experiments, visit the [github page](https://ctallec.github.io/world-models/).

This implementation is based on [ctallec's codebase](https://github.com/ctallec/world-models), where the dependencies are not precisely specified. I overcome this shortcoming by detailing the dependency version and testing it.

Code tested on:

1. Ubuntu22.04 or 20.04
2. Nvidia Driver Version: 545.23.08    CUDA Version: 12.3
3. Python3.9

You'll be safe to go if following this specification.

# Prerequisites

Now you need to install some prerequisites:

First, you need to install `swig` and `xvfb`. On Ubuntu:

```sh
sudo apt-get install swig
sudo apt-get install xvfb
```

Then, you need to install PyTorch in your python environment. Check their website [here](https://pytorch.org) for installation instructions. 

The rest of the requirements is included in the [requirements file](requirements.txt), to install them:

```bash
pip3 install -r requirements.txt
```

* You may notice that in `requirements.txt`, `gym==0.9.4`, which is a rather old version. This is the same version of `gym` in David Ha's original implementation (`gym==0.9.4`). Meanwhile, our code does **NOT** work on gym 0.10.x.
* `pyglet==1.3.2` in order to avoid [this bug](https://stackoverflow.com/questions/56946417/getting-attributeerror-imagedata-object-has-no-attribute-data-in-headle).

At last, you need to create a dir:
```shell
mkdir exp_dir
```

# When running on a headless server

> You may read this chapter if you run on a headless server.

It's common for machine learning algorithms running on a headless server, i.e., the server doesn't connect to a  monitor.

However, running our world model scripts demands the system has a graphical output. And a headless server doesn't have a display.

So, for those scripts needing a graphical output, you can run them with prepending script xvfb-run -s "-screen 0 1400x900x24". That is:

```sh
xvfb-run -s "-screen 0 1400x900x24" python <script>
```

`xvfb-run` will create a X server in memory instead of displaying them on a screen, which is compatitable with headless servers.

But at the testing process, you must watch the grahpic outputs on your local machine. In this case you need to use VNC or x11 forwarding.

I've tested with x11 forwarding and got this error:

```
pyglet.gl.ContextException: Could not create GL context
```

As a result, I have to use VNC.

Note: the test script needs GLX support, so `tightvnc` may not satisfy as it doesn't support GLX. You can freely use `tigervnc` or `x11vnc` or else.

# Training

The model is composed of three parts:

  1. A Variational Auto-Encoder (VAE), whose task is to compress the input images into a compact latent representation.
  2. A Mixture-Density Recurrent Network (MDN-RNN), trained to predict the latent encoding of the next frame given past latent encodings and actions.
  3. A linear Controller (C), which takes both the latent encoding of the current frame, and the hidden state of the MDN-RNN given past latents and actions as input and outputs an action. It is trained to maximize the cumulated reward using the Covariance-Matrix Adaptation Evolution-Strategy ([CMA-ES](http://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaartic.pdf)) from the `cma` python package.

In the given code, all three sections are trained separately, using the scripts `trainvae.py`, `trainmdrnn.py` and `traincontroller.py`.

Training scripts take as argument:
* **--logdir** : The directory in which the models will be stored. If the logdir specified already exists, it loads the old model and continues the training.
* **--noreload** : If you want to override a model in *logdir* instead of reloading it, add this option.

## 1. Data generation

Before launching the VAE and MDN-RNN training scripts, you need to generate a dataset of random rollouts and place it in the `datasets/carracing` folder.

Data generation is handled through the `data/generation_script.py` script, e.g.
```bash
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```

Rollouts are generated using a *brownian* random policy, instead of the *white noise* random `action_space.sample()` policy from gym, providing more consistent rollouts.

> If you're a headless server, you should run:
>
> ```shell
> xvfb-run -s "-screen 0 1400x900x24" python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
> ```

## 2. the VAE

The VAE is trained using the `trainvae.py` file, e.g.
```bash
python trainvae.py --logdir exp_dir
```

## 3. Training the MDN-RNN

The MDN-RNN is trained using the `trainmdrnn.py` file, e.g.
```bash
python trainmdrnn.py --logdir exp_dir
```
A VAE must have been trained in the same `exp_dir` for this script to work.
## 4. Training and testing the Controller

Finally, the controller is trained using CMA-ES, e.g.

```shell
python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```

> If If you're a headless server, you should run:
>
> ```bash
> xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
> ```

The traing of controller is extremely slow. Make sure you have a strong GPU server.

# Testing

You can test the obtained policy with `test_controller.py` e.g.

```bash
python test_controller.py --logdir exp_dir --render
```
* `--render`: render output to your X server.
* To run on a headless server, you should `xvfb-run -s "-screen 0 1400x900x24" python test_controller.py --logdir exp_dir`. But it's meaningless since you won't see any graphic output, you'll only get a test score.
> If If you use a headless server, you should use VNC to connect to thr server. Then run this command. The output will show in your VNC screen.

## Notes

When running on a headless server, you will need to use `xvfb-run` to launch the controller training script. For instance,
```bash
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
```
If you do not have a display available and you launch `traincontroller` without
`xvfb-run`, the script will fail silently (but logs are available in
`logdir/tmp`).

Be aware that `traincontroller` requires heavy gpu memory usage when launched
on gpus. To reduce the memory load, you can directly modify the maximum number
of workers by specifying the `--max-workers` argument.

If you have several GPUs available, `traincontroller` will take advantage of
all gpus specified by `CUDA_VISIBLE_DEVICES`.

## Author

[//]: # (* [Yukuan Lu &#40;陆昱宽&#41;]&#40;https://github.com/LYK-love/World-Models-Pytorch&#41;.)