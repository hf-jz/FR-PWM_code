#!/bin/bash
# FR-PWM C++ Safety Kernel 构建（pybind11，无 CMake 依赖）
set -e
cd "$(dirname "$0")"
PYINC=$(/usr/bin/python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
# CLT python3.9 头文件在 framework Headers（sysconfig include 为空目录时回退）
[ -f "$PYINC/Python.h" ] || PYINC=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Headers
PBINC=$(/usr/bin/python3 -c "import pybind11; print(pybind11.get_include())")
EXT=$(/usr/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
c++ -O2 -std=c++17 -shared -fPIC -undefined dynamic_lookup \
    -I"$PYINC" -I"$PBINC" \
    safety_kernel.cpp -o "../frpwm_cpp$EXT"
echo "built frpwm/cpp -> frpwm/frpwm_cpp$EXT"
