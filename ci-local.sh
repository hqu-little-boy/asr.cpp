#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ci-local.sh  —  Run the GitHub Actions CI workflow locally using Docker.
#
# Usage:
#   ./ci-local.sh           # run all tests (Linux job only, same as CI)
#   ./ci-local.sh --clean   # remove cached vcpkg before running
#
# Requirements:
#   - Docker (running)
#   - git (for submodules)
#
# This script directly invokes Docker (instead of `act`) to avoid the
# act 0.2.88 + Docker 29.x "mkdirat var/run/act" incompatibility.
# The steps executed here mirror .github/workflows/ci.yml : test-linux exactly.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VCPKG_CACHE_DIR="${REPO_ROOT}/.vcpkg-cache-local"

if [[ "${1:-}" == "--clean" ]]; then
    echo "🧹  Removing local vcpkg binary cache..."
    rm -rf "$VCPKG_CACHE_DIR"
fi

mkdir -p "$VCPKG_CACHE_DIR"

echo "🐳  Running CI (Linux / ubuntu:act-latest) in Docker..."
echo "    Repo   : $REPO_ROOT"
echo "    Cache  : $VCPKG_CACHE_DIR"
echo ""

docker run --rm \
    --platform linux/amd64 \
    -v "${REPO_ROOT}:/workspace:ro" \
    -v "${VCPKG_CACHE_DIR}:/vcpkg-cache" \
    -w /workspace \
    catthehacker/ubuntu:act-latest \
    bash -c '
        set -euo pipefail

        # ── system packages ───────────────────────────────────────────────
        echo "::group::Install system packages"
        apt-get update -y -q
        apt-get install -y -q \
            build-essential cmake ninja-build \
            pkg-config libssl-dev libcurl4-openssl-dev \
            zip unzip tar curl git
        echo "::endgroup::"

        # ── vcpkg bootstrap ───────────────────────────────────────────────
        echo "::group::Bootstrap vcpkg"
        export VCPKG_ROOT=/tmp/vcpkg
        export VCPKG_DEFAULT_BINARY_CACHE=/vcpkg-cache
        git clone --depth 1 https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
        "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
        echo "::endgroup::"

        # ── cmake configure ───────────────────────────────────────────────
        echo "::group::Configure CMake"
        cmake -B /tmp/build \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
            -DASR_BUILD_ENGINE=OFF \
            -DASR_BUILD_TESTS=ON \
            -S /workspace
        echo "::endgroup::"

        # ── build ─────────────────────────────────────────────────────────
        echo "::group::Build"
        cmake --build /tmp/build --config Release --parallel "$(nproc)"
        echo "::endgroup::"

        # ── tests ─────────────────────────────────────────────────────────
        echo "::group::Run unit tests"
        cd /tmp/build
        ctest --output-on-failure --build-config Release -j "$(nproc)"
        echo "::endgroup::"
    '

echo ""
echo "✅  All tests passed!"
