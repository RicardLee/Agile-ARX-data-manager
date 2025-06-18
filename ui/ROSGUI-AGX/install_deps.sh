#!/bin/bash

# ROSGUI-AGX System Dependencies Installation Script
# This script installs all required system dependencies for ROSGUI-AGX
# Based on the requirements specified in README.md

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# Check for sudo privileges
if ! sudo -n true 2>/dev/null; then
    print_info "This script requires sudo privileges. You may be prompted for your password."
fi

print_section "1. 更新软件包索引"
print_info "Updating package index..."
sudo apt update

print_section "2. 安装基础编译工具"
print_info "Installing basic build tools..."
sudo apt install -y build-essential cmake pkg-config git wget unzip jq

print_section "3. 添加GCC-13仓库并安装"
print_info "Adding repository for GCC-13..."

# Add the Ubuntu Toolchain PPA for newer GCC versions
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update

print_info "Installing GCC-13 and G++-13..."
sudo apt install -y gcc-13 g++-13

print_info "Setting up symbolic links for GCC and G++..."
sudo ln -sf /usr/bin/gcc-13 /usr/bin/gcc
sudo ln -sf /usr/bin/g++-13 /usr/bin/g++

# Verify GCC version
gcc_version=$(gcc --version | head -n1)
print_success "GCC version: $gcc_version"

print_section "4. 升级CMake到3.20+"
print_info "Checking CMake version..."
cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3 | cut -d. -f1-2)
if dpkg --compare-versions "$cmake_version" "lt" "3.20"; then
    print_warning "CMake version is $cmake_version, upgrading to 3.20+..."
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt update
    sudo apt install -y cmake
    print_success "CMake upgraded successfully"
else
    print_success "CMake version $cmake_version is sufficient"
fi

print_section "5. 安装图形库和OpenGL依赖"
print_info "Installing graphics and OpenGL dependencies..."

# Core graphics dependencies (required)
sudo apt install -y libgl1-mesa-dev libxkbcommon-dev

# X11 support dependencies (required)
sudo apt install -y xorg-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

# Wayland support dependencies (optional, for modern Linux desktops)
sudo apt install -y libwayland-dev wayland-protocols

print_success "Graphics dependencies installed"

print_section "6. 安装GLFW 3.4+"
print_info "Checking GLFW version..."

# Check if GLFW 3.4+ is available from system packages
if pkg-config --exists glfw3; then
    glfw_version=$(pkg-config --modversion glfw3)
    if dpkg --compare-versions "$glfw_version" "ge" "3.4"; then
        print_success "System GLFW version $glfw_version is sufficient"
        sudo apt install -y libglfw3-dev
    else
        print_warning "System GLFW version $glfw_version is too old, will compile from source"
        install_glfw_from_source=true
    fi
else
    print_warning "GLFW not found in system packages, will compile from source"
    install_glfw_from_source=true
fi

if [[ "$install_glfw_from_source" == "true" ]]; then
    print_info "Compiling GLFW 3.4 from source..."
    
    # Remove old versions
    sudo apt remove -y libglfw3-dev libglfw3 2>/dev/null || true
    
    # Download and compile GLFW 3.4
    temp_dir="/tmp/glfw_build_$$"
    mkdir -p "$temp_dir" && cd "$temp_dir"
    
    print_info "Downloading GLFW 3.4..."
    wget -q https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
    unzip -q glfw-3.4.zip && cd glfw-3.4
    
    print_info "Building GLFW..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    
    # Update PKG_CONFIG_PATH
    if ! grep -q "/usr/local/lib/pkgconfig" ~/.bashrc; then
        echo 'export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"' >> ~/.bashrc
    fi
    
    # Clean up
    cd / && rm -rf "$temp_dir"
    
    # Verify installation
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
    installed_version=$(pkg-config --modversion glfw3)
    print_success "GLFW $installed_version compiled and installed successfully"
fi

print_section "7. 安装Python开发环境"
print_info "Installing Python development environment..."
sudo apt install -y python3 python3-pip python3-dev python3-venv
print_success "Python development environment installed"

print_section "8. 安装中文字体支持"
print_info "Installing Chinese font support..."
sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk
print_success "Chinese fonts installed"

print_section "9. 安装内存调试工具 (可选)"
print_info "Installing memory debugging tools (optional but recommended)..."
sudo apt install -y valgrind gdb
print_success "Memory debugging tools installed"

print_section "10. 验证安装"
print_info "Verifying installations..."

echo
print_info "Compiler versions:"
gcc --version | head -n1
g++ --version | head -n1

print_info "Build tools:"
cmake --version | head -n1

print_info "GLFW version:"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
pkg-config --modversion glfw3

print_info "Python version:"
python3 --version

print_info "Verifying C++20 support..."
if echo '#include <concepts>' | gcc -x c++ -std=c++20 -fsyntax-only - 2>/dev/null; then
    print_success "C++20 support: OK"
else
    print_error "C++20 support: FAILED"
fi

print_info "Memory debugging tools:"
if valgrind --version >/dev/null 2>&1; then
    print_success "Valgrind: OK"
else
    print_warning "Valgrind: Not available"
fi

if gdb --version >/dev/null 2>&1; then
    print_success "GDB: OK"
else
    print_warning "GDB: Not available"
fi

print_section "安装完成!"
print_success "All system dependencies have been installed successfully!"
print_info "You can now run './setup.sh' to build the ROSGUI application."
print_warning "Please restart your terminal or run 'source ~/.bashrc' to update your environment variables."

echo
print_info "Next steps:"
echo "  1. Restart your terminal or run: source ~/.bashrc"
echo "  2. Run the setup script: ./setup.sh"
echo "  3. Launch the application: ./run_rosgui.sh" 