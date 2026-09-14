#!/usr/bin/env bash
# Builds Embree 4.4.1 as static libraries for the ray-cast depth backend: one AVX2 code path, no runtime
# dispatch, every hardware reciprocal replaced by IEEE division (exact_division.patch), single-threaded
# tree builds, and a built tree saved and loaded as one image (tree_cache.patch). The compiled copy is kept
# in a cache keyed by Embree version, patches, this script (the flags) and the compiler, and copied into
# prefix/ next to this script; a key already in the cache is never rebuilt. setup.py runs this on every
# build and it returns at once when prefix/ carries the current key.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX_DIR="${SCRIPT_DIR}/prefix"
VERSION=4.4.1
ARCHIVE_SHA256=dcf338cc61b636c871ccf370e673bfd380c5ecb71ce49ad50f28e1d4ec9995dc
CACHE_ROOT="${SWARM_BULLET3_EMBREE_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/swarm-bullet3/embree}"

# The key changes whenever the version, a patch, the flags in this file or the compiler cmake will pick change.
read -r -a cc_cmd <<< "${CC:-cc}"
read -r -a cxx_cmd <<< "${CXX:-c++}"
KEY="$( {
    echo "${VERSION} ${ARCHIVE_SHA256}"
    sha256sum "${SCRIPT_DIR}/exact_division.patch" "${SCRIPT_DIR}/tree_cache.patch" "${BASH_SOURCE[0]}" | cut -d' ' -f1
    "${cc_cmd[@]}" --version | head -n 1
    "${cxx_cmd[@]}" --version | head -n 1
    "${cxx_cmd[@]}" -dumpmachine
} | sha256sum | cut -c1-16)"

if [[ -f "${PREFIX_DIR}/KEY" && "$(cat "${PREFIX_DIR}/KEY")" == "${KEY}" ]]; then
    exit 0
fi

CACHED="${CACHE_ROOT}/${KEY}"
if [[ ! -f "${CACHED}/lib/libembree4.a" ]]; then
    if ! command -v cmake >/dev/null 2>&1; then
        echo "building Embree requires cmake" >&2
        exit 1
    fi

    mkdir -p "${CACHE_ROOT}"
    ARCHIVE="${CACHE_ROOT}/embree-v${VERSION}.tar.gz"
    if [[ ! -f "${ARCHIVE}" ]]; then
        curl --fail --location --retry 3 --output "${ARCHIVE}" \
            "https://github.com/RenderKit/embree/archive/refs/tags/v${VERSION}.tar.gz"
    fi
    echo "${ARCHIVE_SHA256}  ${ARCHIVE}" | sha256sum --check --quiet

    # Source and build trees live under one fixed path per key, so a rebuild of the same key gives the same bytes.
    WORK_DIR="${CACHE_ROOT}/build-${KEY}"
    SOURCE_DIR="${WORK_DIR}/src"
    BUILD_DIR="${WORK_DIR}/build"
    INSTALL_DIR="${WORK_DIR}/install"
    rm -rf "${WORK_DIR}"
    mkdir -p "${SOURCE_DIR}"
    tar -xzf "${ARCHIVE}" --strip-components=1 -C "${SOURCE_DIR}"
    patch -d "${SOURCE_DIR}" -p1 --quiet < "${SCRIPT_DIR}/exact_division.patch"
    # Embree offers no way to save or read back a built tree, so the two calls that do it are added here.
    patch -d "${SOURCE_DIR}" -p1 --quiet < "${SCRIPT_DIR}/tree_cache.patch"

    # The patch must leave no approximate reciprocal or inverse square root in any code that gets compiled.
    if grep -rnE '_mm(256|512)?_(rcp14|rsqrt14|rcp|rsqrt)_[sp]s' --include='*.h' --include='*.cpp' \
        "${SOURCE_DIR}/common" "${SOURCE_DIR}/kernels" | grep -v '/arm/'; then
        echo "Embree still contains a hardware reciprocal after patching" >&2
        exit 1
    fi

    # One ISA compiled in and MAX_ISA=NONE: the library holds a single AVX2 code path and never dispatches.
    cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
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

    if [[ ! -f "${INSTALL_DIR}/lib/libembree4.a" ]]; then
        echo "Embree build did not produce ${INSTALL_DIR}/lib/libembree4.a" >&2
        exit 1
    fi
    # The cache entry appears in one move, so an interrupted build never leaves a half-written key behind.
    rm -rf "${CACHED}"
    mv "${INSTALL_DIR}" "${CACHED}"
    rm -rf "${WORK_DIR}"
fi

rm -rf "${PREFIX_DIR}"
cp -r "${CACHED}" "${PREFIX_DIR}"
echo "${KEY}" > "${PREFIX_DIR}/KEY"
