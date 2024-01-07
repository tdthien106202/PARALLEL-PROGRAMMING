# Lenet5

## Usage
Download and unzip [fashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset in `Lenet5_PP/data/mnist/`.

```shell
#ran on google colab
!git clone https://github.com/NLTDat20CLC/Lenet5_PP.git
%cd content/Lenet5_PP
%cd data/mnist
%cd /content/Lenet5_PP
!mkdir build
%cd build
!cmake .. -DCMAKE_{C,CXX}_FLAGS="-O3 -march=native"
!make -j$(nproc)
```

Run `./demo`.

Result: 
