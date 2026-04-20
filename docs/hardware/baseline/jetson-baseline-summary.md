# Jetson Orin Baseline Summary

Generated on: Sun 19 Apr 2026 03:26:53 PM IST

## Hostname
```
 Static hostname: mm-orin
       Icon name: computer
      Machine ID: 5dbfb12414a3456d9014d88183e338b1
         Boot ID: bca16627b44648fa8364554cefdc6aff
Operating System: Ubuntu 22.04.5 LTS
          Kernel: Linux 5.15.148-tegra
    Architecture: arm64
 Hardware Vendor: NVIDIA
  Hardware Model: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
```

## OS Release
```
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

## Kernel / Architecture
```
Linux mm-orin 5.15.148-tegra #1 SMP PREEMPT Thu Sep 18 15:08:33 PDT 2025 aarch64 aarch64 aarch64 GNU/Linux
```

## CPU Information
```
Architecture:                       aarch64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
CPU(s):                             6
On-line CPU(s) list:                0-5
Vendor ID:                          ARM
Model name:                         Cortex-A78AE
Model:                              1
Thread(s) per core:                 1
Core(s) per cluster:                3
Socket(s):                          -
Cluster(s):                         2
Stepping:                           r0p1
CPU max MHz:                        1728.0000
CPU min MHz:                        115.2000
BogoMIPS:                           62.50
Flags:                              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp uscat ilrcpc flagm paca pacg
L1d cache:                          384 KiB (6 instances)
L1i cache:                          384 KiB (6 instances)
L2 cache:                           1.5 MiB (6 instances)
L3 cache:                           4 MiB (2 instances)
NUMA node(s):                       1
NUMA node0 CPU(s):                  0-5
Vulnerability Gather data sampling: Not affected
Vulnerability Itlb multihit:        Not affected
Vulnerability L1tf:                 Not affected
Vulnerability Mds:                  Not affected
Vulnerability Meltdown:             Not affected
Vulnerability Mmio stale data:      Not affected
Vulnerability Retbleed:             Not affected
Vulnerability Spec rstack overflow: Not affected
Vulnerability Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:           Mitigation; __user pointer sanitization
Vulnerability Spectre v2:           Mitigation; CSV2, BHB
Vulnerability Srbds:                Not affected
Vulnerability Tsx async abort:      Not affected
```

## Memory
```
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       1.3Gi       3.9Gi        22Mi       2.3Gi       5.9Gi
Swap:          3.7Gi          0B       3.7Gi
```

## Disk Layout
```
NAME         MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0          7:0    0    16M  1 loop 
mmcblk0      179:0    0 119.4G  0 disk 
├─mmcblk0p1  179:1    0 117.9G  0 part /
├─mmcblk0p2  179:2    0   128M  0 part 
├─mmcblk0p3  179:3    0   768K  0 part 
├─mmcblk0p4  179:4    0  31.6M  0 part 
├─mmcblk0p5  179:5    0   128M  0 part 
├─mmcblk0p6  179:6    0   768K  0 part 
├─mmcblk0p7  179:7    0  31.6M  0 part 
├─mmcblk0p8  179:8    0    80M  0 part 
├─mmcblk0p9  179:9    0   512K  0 part 
├─mmcblk0p10 179:10   0    64M  0 part /boot/efi
├─mmcblk0p11 179:11   0    80M  0 part 
├─mmcblk0p12 179:12   0   512K  0 part 
├─mmcblk0p13 179:13   0    64M  0 part 
├─mmcblk0p14 179:14   0   400M  0 part 
└─mmcblk0p15 179:15   0 479.5M  0 part 
zram0        252:0    0   635M  0 disk [SWAP]
zram1        252:1    0   635M  0 disk [SWAP]
zram2        252:2    0   635M  0 disk [SWAP]
zram3        252:3    0   635M  0 disk [SWAP]
zram4        252:4    0   635M  0 disk [SWAP]
zram5        252:5    0   635M  0 disk [SWAP]
```

## Disk Usage
```
Filesystem       Size  Used Avail Use% Mounted on
/dev/mmcblk0p1   116G   20G   92G  18% /
tmpfs            3.8G  120K  3.8G   1% /dev/shm
tmpfs            1.5G   19M  1.5G   2% /run
tmpfs            5.0M  4.0K  5.0M   1% /run/lock
/dev/mmcblk0p10   63M  110K   63M   1% /boot/efi
tmpfs            762M  132K  762M   1% /run/user/1000
```

## Jetson Power Model
```
NV Power Mode: 25W
1
```

## Jetson Clocks
```
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
Online CPUs: 0-5
cpu0:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1190400 IdleStates: WFI=1 c7=1 
cpu1:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1344000 IdleStates: WFI=1 c7=1 
cpu2:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=729600 IdleStates: WFI=1 c7=1 
cpu3:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1190400 IdleStates: WFI=1 c7=1 
cpu4:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=1113600 IdleStates: WFI=1 c7=1 
cpu5:  Online=1 Governor=schedutil MinFreq=729600 MaxFreq=1344000 CurrentFreq=729600 IdleStates: WFI=1 c7=1 
GPU MinFreq=306000000 MaxFreq=918000000 CurrentFreq=306000000
Active GPU TPCs: 4
EMC MinFreq=204000000 MaxFreq=3199000000 CurrentFreq=665600000 FreqOverride=0
FAN Dynamic Speed Control=nvfancontrol hwmon0_pwm1=58
NV Power Mode: 25W
```

## Boot Slots
```
Current version: 36.4.7
Capsule update status: 1
Current bootloader slot: A
Active bootloader slot: A
num_slots: 2
slot: 0,             status: normal
slot: 1,             status: normal
```

## CUDA Version
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

## Python Version
```
Python 3.10.12
```

## Installed CUDA Packages
```
ii  cuda-cccl-12-6                                           12.6.37-1                                        arm64        CUDA CCCL
ii  cuda-command-line-tools-12-6                             12.6.11-1                                        arm64        CUDA command-line tools
ii  cuda-compiler-12-6                                       12.6.11-1                                        arm64        CUDA compiler
ii  cuda-crt-12-6                                            12.6.68-1                                        arm64        CUDA crt
ii  cuda-cudart-12-6                                         12.6.68-1                                        arm64        CUDA Runtime native Libraries
ii  cuda-cudart-dev-12-6                                     12.6.68-1                                        arm64        CUDA Runtime native dev links, headers
ii  cuda-cuobjdump-12-6                                      12.6.68-1                                        arm64        CUDA cuobjdump
ii  cuda-cupti-12-6                                          12.6.68-1                                        arm64        CUDA profiling tools runtime libs.
ii  cuda-cupti-dev-12-6                                      12.6.68-1                                        arm64        CUDA profiling tools interface.
ii  cuda-cuxxfilt-12-6                                       12.6.68-1                                        arm64        CUDA cuxxfilt
ii  cuda-documentation-12-6                                  12.6.68-1                                        arm64        CUDA documentation
ii  cuda-driver-dev-12-6                                     12.6.68-1                                        arm64        CUDA Driver native dev stub library
ii  cuda-gdb-12-6                                            12.6.68-1                                        arm64        CUDA-GDB
ii  cuda-libraries-12-6                                      12.6.11-1                                        arm64        CUDA Libraries 12.6 meta-package
ii  cuda-libraries-dev-12-6                                  12.6.11-1                                        arm64        CUDA Libraries 12.6 development meta-package
ii  cuda-nsight-compute-12-6                                 12.6.11-1                                        arm64        NVIDIA Nsight Compute
ii  cuda-nvcc-12-6                                           12.6.68-1                                        arm64        CUDA nvcc
ii  cuda-nvdisasm-12-6                                       12.6.68-1                                        arm64        CUDA disassembler
ii  cuda-nvml-dev-12-6                                       12.6.68-1                                        arm64        NVML native dev links, headers
ii  cuda-nvprune-12-6                                        12.6.68-1                                        arm64        CUDA nvprune
ii  cuda-nvrtc-12-6                                          12.6.68-1                                        arm64        NVRTC native runtime libraries
ii  cuda-nvrtc-dev-12-6                                      12.6.68-1                                        arm64        NVRTC native dev links, headers
ii  cuda-nvtx-12-6                                           12.6.68-1                                        arm64        NVIDIA Tools Extension
ii  cuda-nvvm-12-6                                           12.6.68-1                                        arm64        CUDA nvvm
ii  cuda-profiler-api-12-6                                   12.6.68-1                                        arm64        CUDA Profiler API
ii  cuda-sanitizer-12-6                                      12.6.68-1                                        arm64        CUDA Sanitizer
ii  cuda-toolkit-12                                          12.6.11-1                                        arm64        CUDA Toolkit 12 meta-package
ii  cuda-toolkit-12-6                                        12.6.11-1                                        arm64        CUDA Toolkit 12.6 meta-package
ii  cuda-toolkit-12-6-config-common                          12.6.68-1                                        all          Common config package for CUDA Toolkit 12.6.
ii  cuda-toolkit-12-config-common                            12.6.68-1                                        all          Common config package for CUDA Toolkit 12.
ii  cuda-toolkit-config-common                               12.6.68-1                                        all          Common config package for CUDA Toolkit.
ii  cuda-tools-12-6                                          12.6.11-1                                        arm64        CUDA Tools meta-package
ii  cuda-visual-tools-12-6                                   12.6.11-1                                        arm64        CUDA visual tools
ii  l4t-cuda-tegra-repo-ubuntu2204-12-6-local                12.6.11-1                                        arm64        l4t-cuda-tegra repository configuration files
ii  libcudnn9-cuda-12                                        9.3.0.75-1                                       arm64        cuDNN runtime libraries for CUDA 12.6
ii  libcudnn9-dev-cuda-12                                    9.3.0.75-1                                       arm64        cuDNN development headers and symlinks for CUDA 12.6
ii  libcudnn9-static-cuda-12                                 9.3.0.75-1                                       arm64        cuDNN static libraries for CUDA 12.6
ii  libnvinfer-bin                                           10.3.0.30-1+cuda12.5                             arm64        TensorRT binaries
ii  libnvinfer-dev                                           10.3.0.30-1+cuda12.5                             arm64        TensorRT development libraries
ii  libnvinfer-dispatch-dev                                  10.3.0.30-1+cuda12.5                             arm64        TensorRT development dispatch runtime libraries
ii  libnvinfer-dispatch10                                    10.3.0.30-1+cuda12.5                             arm64        TensorRT dispatch runtime library
ii  libnvinfer-headers-dev                                   10.3.0.30-1+cuda12.5                             arm64        TensorRT development headers
ii  libnvinfer-headers-plugin-dev                            10.3.0.30-1+cuda12.5                             arm64        TensorRT plugin headers
ii  libnvinfer-lean-dev                                      10.3.0.30-1+cuda12.5                             arm64        TensorRT lean runtime libraries
ii  libnvinfer-lean10                                        10.3.0.30-1+cuda12.5                             arm64        TensorRT lean runtime library
ii  libnvinfer-plugin-dev                                    10.3.0.30-1+cuda12.5                             arm64        TensorRT plugin libraries
ii  libnvinfer-plugin10                                      10.3.0.30-1+cuda12.5                             arm64        TensorRT plugin libraries
ii  libnvinfer-samples                                       10.3.0.30-1+cuda12.5                             all          TensorRT samples
ii  libnvinfer-vc-plugin-dev                                 10.3.0.30-1+cuda12.5                             arm64        TensorRT vc-plugin library
ii  libnvinfer-vc-plugin10                                   10.3.0.30-1+cuda12.5                             arm64        TensorRT vc-plugin library
ii  libnvinfer10                                             10.3.0.30-1+cuda12.5                             arm64        TensorRT runtime libraries
ii  libnvonnxparsers-dev                                     10.3.0.30-1+cuda12.5                             arm64        TensorRT ONNX libraries
ii  libnvonnxparsers10                                       10.3.0.30-1+cuda12.5                             arm64        TensorRT ONNX libraries
ii  nv-tensorrt-local-tegra-repo-ubuntu2204-10.3.0-cuda-12.5 1.0-1                                            arm64        nv-tensorrt-local-tegra repository configuration files
ii  nvidia-l4t-cuda                                          36.4.7-20250918154033                            arm64        NVIDIA CUDA Package
ii  nvidia-l4t-cuda-utils                                    36.4.7-20250918154033                            arm64        NVIDIA CUDA utilities
ii  nvidia-l4t-cudadebuggingsupport                          12.6-34622040.0                                  arm64        NVIDIA CUDA Debugger Support Package
ii  python3-libnvinfer                                       10.3.0.30-1+cuda12.5                             arm64        Python 3 bindings for TensorRT standard runtime
ii  python3-libnvinfer-dev                                   10.3.0.30-1+cuda12.5                             arm64        Python 3 development package for TensorRT standard runtime
ii  python3-libnvinfer-dispatch                              10.3.0.30-1+cuda12.5                             arm64        Python 3 bindings for TensorRT dispatch runtime
ii  python3-libnvinfer-lean                                  10.3.0.30-1+cuda12.5                             arm64        Python 3 bindings for TensorRT lean runtime
ii  tensorrt                                                 10.3.0.30-1+cuda12.5                             arm64        Meta package for TensorRT
```

## Installed cuDNN Packages
```
ii  cudnn-local-tegra-repo-ubuntu2204-9.3.0                  1.0-1                                            arm64        cudnn-local-tegra repository configuration files
ii  libcudnn9-cuda-12                                        9.3.0.75-1                                       arm64        cuDNN runtime libraries for CUDA 12.6
ii  libcudnn9-dev-cuda-12                                    9.3.0.75-1                                       arm64        cuDNN development headers and symlinks for CUDA 12.6
ii  libcudnn9-samples                                        9.3.0.75-1                                       all          cuDNN samples
ii  libcudnn9-static-cuda-12                                 9.3.0.75-1                                       arm64        cuDNN static libraries for CUDA 12.6
```

## Installed TensorRT Packages
```
ii  libnvinfer-bin                                           10.3.0.30-1+cuda12.5                             arm64        TensorRT binaries
ii  libnvinfer-dev                                           10.3.0.30-1+cuda12.5                             arm64        TensorRT development libraries
ii  libnvinfer-dispatch-dev                                  10.3.0.30-1+cuda12.5                             arm64        TensorRT development dispatch runtime libraries
ii  libnvinfer-dispatch10                                    10.3.0.30-1+cuda12.5                             arm64        TensorRT dispatch runtime library
ii  libnvinfer-headers-dev                                   10.3.0.30-1+cuda12.5                             arm64        TensorRT development headers
ii  libnvinfer-headers-plugin-dev                            10.3.0.30-1+cuda12.5                             arm64        TensorRT plugin headers
ii  libnvinfer-lean-dev                                      10.3.0.30-1+cuda12.5                             arm64        TensorRT lean runtime libraries
ii  libnvinfer-lean10                                        10.3.0.30-1+cuda12.5                             arm64        TensorRT lean runtime library
ii  libnvinfer-plugin-dev                                    10.3.0.30-1+cuda12.5                             arm64        TensorRT plugin libraries
ii  libnvinfer-plugin10                                      10.3.0.30-1+cuda12.5                             arm64        TensorRT plugin libraries
ii  libnvinfer-samples                                       10.3.0.30-1+cuda12.5                             all          TensorRT samples
ii  libnvinfer-vc-plugin-dev                                 10.3.0.30-1+cuda12.5                             arm64        TensorRT vc-plugin library
ii  libnvinfer-vc-plugin10                                   10.3.0.30-1+cuda12.5                             arm64        TensorRT vc-plugin library
ii  libnvinfer10                                             10.3.0.30-1+cuda12.5                             arm64        TensorRT runtime libraries
ii  libnvonnxparsers-dev                                     10.3.0.30-1+cuda12.5                             arm64        TensorRT ONNX libraries
ii  libnvonnxparsers10                                       10.3.0.30-1+cuda12.5                             arm64        TensorRT ONNX libraries
ii  python3-libnvinfer                                       10.3.0.30-1+cuda12.5                             arm64        Python 3 bindings for TensorRT standard runtime
ii  python3-libnvinfer-dev                                   10.3.0.30-1+cuda12.5                             arm64        Python 3 development package for TensorRT standard runtime
ii  python3-libnvinfer-dispatch                              10.3.0.30-1+cuda12.5                             arm64        Python 3 bindings for TensorRT dispatch runtime
ii  python3-libnvinfer-lean                                  10.3.0.30-1+cuda12.5                             arm64        Python 3 bindings for TensorRT lean runtime
ii  tensorrt                                                 10.3.0.30-1+cuda12.5                             arm64        Meta package for TensorRT
```

## Network Devices
```
DEVICE            TYPE      STATE                   CONNECTION              
enP8p1s0          ethernet  connected               Wired connection 1      
wlP1p1s0          wifi      connected               Airtel_amit_5354_5GHz 1 
docker0           bridge    connected (externally)  docker0                 
p2p-dev-wlP1p1s0  wifi-p2p  disconnected            --                      
l4tbr0            bridge    unmanaged               --                      
can0              can       unmanaged               --                      
usb0              ethernet  unmanaged               --                      
usb1              ethernet  unmanaged               --                      
lo                loopback  unmanaged               --                      
```

## IP Addresses
```
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN group default qlen 10
    link/can 
3: wlP1p1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 50:2e:91:27:f8:15 brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.53/24 brd 192.168.1.255 scope global dynamic noprefixroute wlP1p1s0
       valid_lft 85329sec preferred_lft 85329sec
    inet6 fe80::8a2a:42a:823d:a77c/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
4: l4tbr0: <BROADCAST,MULTICAST> mtu 1500 qdisc noop state DOWN group default qlen 1000
    link/ether 3e:c7:e7:79:f3:87 brd ff:ff:ff:ff:ff:ff
5: usb0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast master l4tbr0 state DOWN group default qlen 1000
    link/ether e2:71:1b:6d:eb:e5 brd ff:ff:ff:ff:ff:ff
6: usb1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast master l4tbr0 state DOWN group default qlen 1000
    link/ether e2:71:1b:6d:eb:e7 brd ff:ff:ff:ff:ff:ff
7: enP8p1s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
    link/ether 4c:bb:47:7b:21:e8 brd ff:ff:ff:ff:ff:ff
    inet 192.168.1.52/24 brd 192.168.1.255 scope global dynamic noprefixroute enP8p1s0
       valid_lft 85328sec preferred_lft 85328sec
    inet6 fe80::ba23:8ead:8f6b:c363/64 scope link noprefixroute 
       valid_lft forever preferred_lft forever
8: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default 
    link/ether 7e:5d:b2:73:6c:ee brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft forever
```

## USB Devices
```
Bus 002 Device 002: ID 0bda:0489 Realtek Semiconductor Corp. 4-Port USB 3.0 Hub
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 001 Device 003: ID 13d3:3549 IMC Networks Bluetooth Radio
Bus 001 Device 005: ID 1a2c:212a China Resource Semico Co., Ltd USB Keyboard
Bus 001 Device 004: ID 1c4f:0034 SiGma Micro XM102K Optical Wheel Mouse
Bus 001 Device 002: ID 0bda:5489 Realtek Semiconductor Corp. 4-Port USB 2.0 Hub
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
```

## PCI Devices
```
0001:00:00.0 PCI bridge: NVIDIA Corporation Device 229e (rev a1)
0001:01:00.0 Network controller: Realtek Semiconductor Co., Ltd. RTL8822CE 802.11ac PCIe Wireless Network Adapter
0008:00:00.0 PCI bridge: NVIDIA Corporation Device 229c (rev a1)
0008:01:00.0 Ethernet controller: Realtek Semiconductor Co., Ltd. RTL8111/8168/8411 PCI Express Gigabit Ethernet Controller (rev 15)
```

## Camera Devices
```
Cannot open device /dev/video0, exiting.
NVIDIA Tegra Video Input Device (platform:tegra-camrtc-ca):
	/dev/media0

ls: cannot access '/dev/video*': No such file or directory
```

## Failed Services
```
  UNIT                      LOAD   ACTIVE SUB    DESCRIPTION
● apport-autoreport.service loaded failed failed Process error reports when automatic reporting is enabled

LOAD   = Reflects whether the unit definition was properly loaded.
ACTIVE = The high-level unit activation state, i.e. generalization of SUB.
SUB    = The low-level unit activation state, values depend on unit type.
1 loaded units listed.
```

