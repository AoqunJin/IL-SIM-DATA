# An imitation learning data collection pipeline for simulators.

## Installation

### MetaWorld

#### Install Mujoco

``` bash
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco/mujoco210
tar -xzvf ./mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
```

Add this to `.bashrc`:

``` bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```
You can avoid error when compiling the cython file with:

``` bash
apt-get install libghc-x11-dev libglew-dev patchelf
pip install Cython==3.0.0a10
```

#### Install MetaWorld

``` bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
```

### RLBench

#### Install for Headless

```bash
apt-get install xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev
# Leave out --use-display-device=None if the GPU is headless, i.e. if it has no display outputs.
nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
echo -e 'Section "ServerFlags"\n\tOption "MaxClients" "2048"\nEndSection\n' \
    | sudo tee /etc/X11/xorg.conf.d/99-maxclients.conf
# Install VirtualGL
wget https://sourceforge.net/projects/virtualgl/files/2.5.2/virtualgl_2.5.2_amd64.deb/download -O virtualgl_2.5.2_amd64.deb
dpkg -i virtualgl*.deb
rm virtualgl*.deb
```

You will now need to reboot, and then start the X server:

``` bash
sudo reboot
nohup sudo X &
```

You can test with:

``` bash
export DISPLAY=:0.0
glxgears
```

#### Install CoppeliaSim

Download CoppeliaSim:

- [Ubuntu 18.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

Add to `.bashrc`:

``` bash
export COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

#### Install PyRep

``` bash
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
```
#### Install RLBench

``` bash
git clone https://github.com/AoqunJin/RLBench.git
cd RLBench
pip install -r requirements.txt
pip install .
```

## Data Collection

### MetaWorld

### RLBench

## Contact Us
Aoqun Jin:

## Citation

## License
