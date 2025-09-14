apt update
apt install -y lsb-release wget software-properties-common gnupg
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100
update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 100
