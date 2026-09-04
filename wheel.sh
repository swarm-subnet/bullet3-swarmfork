#!/bin/bash
set -e -x

# use this docker command
# sudo docker run -it -v $(pwd):/io quay.io/pypa/manylinux2014_x86_64

# x86-64-v3 (AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE: Intel Haswell 2013+, AMD Zen) is the published floor: faster than -O2 with bit-identical output.
export SWARM_BULLET3_OPT_LEVEL=v3
export SWARM_BULLET3_PGO=off

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -r /io/dev/bullet3/requirements.txt
    "${PYBIN}/pip" wheel /io/dev/bullet3 -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl"  -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
done

