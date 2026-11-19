#!/usr/bin/env python
"""Google Colab 环境准备脚本（在仓库根目录执行）。

用法:
    python colab_setup.py --build-diffvg        # 编译 diffvg（优先 CUDA，失败自动回退 CPU）
    python colab_setup.py --download-weights    # 下载 U2Net 前景分割权重
    python colab_setup.py --download-clip       # 预下载 CLIP 两种骨干权重
    python colab_setup.py --verify              # 检查环境是否就绪
    python colab_setup.py --all                 # 以上全部
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DIFFVG_DIR = os.path.join(ROOT, "diffvg")
THRUST_DIR = os.path.join(DIFFVG_DIR, "thrust")
THRUST_TAG = "2.2.0"
WEIGHTS_PATH = os.path.join(ROOT, "U2Net_", "saved_models", "u2net_enhanced_best.pth")
U2NET_GDRIVE_ID = "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
MIN_WEIGHT_BYTES = 100 * 1024 * 1024


def ensure_thrust():
    """CPU 版 diffvg 需要 Thrust 头文件，仓库以 tar.gz 形式内置，首次构建时解压。"""
    if os.path.isfile(os.path.join(THRUST_DIR, "ThrustConfig.cmake")):
        return True

    archive = os.path.join(DIFFVG_DIR, "thrust-%s.tar.gz" % THRUST_TAG)
    if not os.path.isfile(archive):
        print("[thrust] archive not found:", archive, flush=True)
        return False

    print("[thrust] extracting %s ..." % archive, flush=True)
    try:
        import tarfile
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(DIFFVG_DIR)
    except Exception as exc:
        print("[thrust] extract failed:", exc, flush=True)
        return False
    return os.path.isfile(os.path.join(THRUST_DIR, "ThrustConfig.cmake"))


def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def diffvg_importable():
    code = (
        "import sys; sys.path.insert(0, %r); import pydiffvg; "
        "print('pydiffvg:', pydiffvg.__file__)" % DIFFVG_DIR
    )
    return subprocess.run([sys.executable, "-c", code]).returncode == 0


def build_diffvg():
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not ensure_thrust():
        print("[diffvg] Thrust headers are unavailable; CPU build will fail.", flush=True)

    modes = [1, 0] if has_cuda else [0]
    for cuda in modes:
        build_dir = os.path.join(DIFFVG_DIR, "build")
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
        mode_env = env.copy()
        mode_env["DIFFVG_CUDA"] = str(cuda)
        print("\n=== Building diffvg (CUDA=%d) ===" % cuda, flush=True)
        rc = run([sys.executable, "setup.py", "build_ext", "--inplace"],
                 cwd=DIFFVG_DIR, env=mode_env)
        if rc == 0 and diffvg_importable():
            print("\n[diffvg] build succeeded (CUDA=%d)" % cuda, flush=True)
            return True
        print("[diffvg] build failed (CUDA=%d)" % cuda, flush=True)

    print(
        "\n[diffvg] 编译失败。请把上面的编译日志发出来排查；"
        "常见原因是 Colab 的 CMake/CUDA 版本过新。", flush=True)
    return False


def download_weights():
    if os.path.isfile(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) >= MIN_WEIGHT_BYTES:
        print("[weights] already exists:", WEIGHTS_PATH, flush=True)
        return True

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    try:
        import gdown
    except ImportError:
        if run([sys.executable, "-m", "pip", "install", "-q", "gdown"]) != 0:
            return False
        import gdown

    url = "https://drive.google.com/uc?id=" + U2NET_GDRIVE_ID
    print("[weights] downloading U2Net weights (~219MB)...", flush=True)
    try:
        gdown.download(url, WEIGHTS_PATH, quiet=False)
    except Exception as exc:
        print("[weights] download failed:", exc, flush=True)
        return False

    if not os.path.isfile(WEIGHTS_PATH) or os.path.getsize(WEIGHTS_PATH) < MIN_WEIGHT_BYTES:
        print("[weights] downloaded file is incomplete:", WEIGHTS_PATH, flush=True)
        return False
    print("[weights] saved to", WEIGHTS_PATH, flush=True)
    return True


def download_clip():
    sys.path.insert(0, ROOT)
    try:
        import CLIP_.clip as clip
    except Exception as exc:
        print("[clip] import failed:", exc, flush=True)
        return False

    ok = True
    for name in ("RN101", "ViT-B/32"):
        try:
            print("[clip] loading", name, flush=True)
            clip.load(name, device="cpu", jit=False)
        except Exception as exc:
            print("[clip] failed to load %s: %s" % (name, exc), flush=True)
            ok = False
    return ok


def verify():
    ok = True
    print("python:", sys.version.split()[0], flush=True)
    print("repo:", ROOT, flush=True)

    try:
        import torch
        print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0), flush=True)
    except Exception as exc:
        print("torch: FAILED -", exc, flush=True)
        ok = False

    if diffvg_importable():
        print("diffvg: OK", flush=True)
    else:
        print("diffvg: FAILED (run: python colab_setup.py --build-diffvg)", flush=True)
        ok = False

    if os.path.isfile(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) >= MIN_WEIGHT_BYTES:
        print("u2net weights: OK", flush=True)
    else:
        print("u2net weights: MISSING (run: python colab_setup.py --download-weights)", flush=True)
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-diffvg", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--download-clip", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.build_diffvg, args.download_weights,
                args.download_clip, args.verify, args.all]):
        parser.print_help()
        return 0

    ok = True
    if args.all or args.build_diffvg:
        ok = build_diffvg() and ok
    if args.all or args.download_weights:
        ok = download_weights() and ok
    if args.all or args.download_clip:
        ok = download_clip() and ok
    if args.all or args.verify:
        ok = verify() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
