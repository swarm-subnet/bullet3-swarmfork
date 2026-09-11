#!/usr/bin/env bash
# Builds Embree 4.4.1 as static libraries for the ray-cast depth backend: one AVX2 code path, no runtime
# dispatch, every hardware reciprocal replaced by IEEE division (exact_division.patch), single-threaded
# tree builds. Output lands in prefix/ next to this script; setup.py runs it when that folder is missing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/src"
BUILD_DIR="${SCRIPT_DIR}/build"
PREFIX_DIR="${SCRIPT_DIR}/prefix"
LIBRARY="${PREFIX_DIR}/lib/libembree4.a"
VERSION=4.4.1
ARCHIVE_SHA256=dcf338cc61b636c871ccf370e673bfd380c5ecb71ce49ad50f28e1d4ec9995dc

if [[ -f "${LIBRARY}" ]]; then
    exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "building Embree requires cmake" >&2
    exit 1
fi

ARCHIVE="${SCRIPT_DIR}/embree-v${VERSION}.tar.gz"
if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 --output "${ARCHIVE}" \
        "https://github.com/RenderKit/embree/archive/refs/tags/v${VERSION}.tar.gz"
fi
echo "${ARCHIVE_SHA256}  ${ARCHIVE}" | sha256sum --check --quiet

rm -rf "${SOURCE_DIR}" "${BUILD_DIR}" "${PREFIX_DIR}"
mkdir -p "${SOURCE_DIR}"
tar -xzf "${ARCHIVE}" --strip-components=1 -C "${SOURCE_DIR}"
patch -d "${SOURCE_DIR}" -p1 --quiet < "${SCRIPT_DIR}/exact_division.patch"

# The patch must leave no approximate reciprocal or inverse square root in any code that gets compiled.
if grep -rnE '_mm(256|512)?_(rcp14|rsqrt14|rcp|rsqrt)_[sp]s' --include='*.h' --include='*.cpp' \
    "${SOURCE_DIR}/common" "${SOURCE_DIR}/kernels" | grep -v '/arm/'; then
    echo "Embree still contains a hardware reciprocal after patching" >&2
    exit 1
fi

# One ISA compiled in and MAX_ISA=NONE: the library holds a single AVX2 code path and never dispatches.
cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX_DIR}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DEMBREE_STATIC_LIB=ON \
    -DEMBREE_ISPC_SUPPORT=OFF \
    -DEMBREE_TUTORIALS=OFF \
    -DEMBREE_TASKING_SYSTEM=INTERNAL \
    -DEMBREE_MAX_ISA=NONE \
    -DEMBREE_ISA_SSE2=OFF \
    -DEMBREE_ISA_SSE42=OFF \
    -DEMBREE_ISA_AVX=OFF \
    -DEMBREE_ISA_AVX2=ON \
    -DEMBREE_ISA_AVX512=OFF \
    -DEMBREE_GEOMETRY_TRIANGLE=ON \
    -DEMBREE_GEOMETRY_QUAD=OFF \
    -DEMBREE_GEOMETRY_CURVE=OFF \
    -DEMBREE_GEOMETRY_SUBDIVISION=OFF \
    -DEMBREE_GEOMETRY_USER=OFF \
    -DEMBREE_GEOMETRY_INSTANCE=ON \
    -DEMBREE_GEOMETRY_INSTANCE_ARRAY=OFF \
    -DEMBREE_GEOMETRY_GRID=OFF \
    -DEMBREE_GEOMETRY_POINT=OFF \
    -DEMBREE_FILTER_FUNCTION=ON \
    -DEMBREE_RAY_MASK=OFF \
    -DEMBREE_BACKFACE_CULLING=OFF \
    -DEMBREE_COMPACT_POLYS=OFF \
    -DEMBREE_SYCL_SUPPORT=OFF > "${SCRIPT_DIR}/configure.log"
cmake --build "${BUILD_DIR}" -j"$(nproc)" > "${SCRIPT_DIR}/build.log"
cmake --install "${BUILD_DIR}" > /dev/null

if [[ ! -f "${LIBRARY}" ]]; then
    echo "Embree build did not produce ${LIBRARY}" >&2
    exit 1
fi
