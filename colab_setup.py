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
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DIFFVG_DIR = os.path.join(ROOT, "diffvg")
WEIGHTS_PATH = os.path.join(ROOT, "U2Net_", "saved_models", "u2net_enhanced_best.pth")
# Enhanced 权重无法直接提交到 git（GitHub 单文件上限 100MB），因此切成
# 3 个 <100MB 的分片放在仓库里，下载后拼接并校验 MD5。
U2NET_PARTS = ["u2net_enhanced_best.part-00",
               "u2net_enhanced_best.part-01",
               "u2net_enhanced_best.part-02"]
U2NET_PARTS_BASE = ("https://raw.githubusercontent.com/tuanzi091/"
                    "clipasso-colab/main/U2Net_/saved_models/parts/")
U2NET_BYTES = 219241823
U2NET_MD5 = "4DEFD8218475618E901F4DCF41FDE3EE"


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
            mode_file = os.path.join(DIFFVG_DIR, "pydiffvg", "build_mode.txt")
            with open(mode_file, "w") as handle:
                handle.write("cuda" if cuda else "cpu")
            print("\n[diffvg] build succeeded (CUDA=%d)" % cuda, flush=True)
            return True
        print("[diffvg] build failed (CUDA=%d)" % cuda, flush=True)

    print(
        "\n[diffvg] 编译失败。请把上面的编译日志发出来排查；"
        "常见原因是 Colab 的 CMake/CUDA 版本过新。", flush=True)
    return False


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def weights_ready():
    return (os.path.isfile(WEIGHTS_PATH)
            and os.path.getsize(WEIGHTS_PATH) == U2NET_BYTES
            and _md5(WEIGHTS_PATH) == U2NET_MD5)


def download_weights():
    if weights_ready():
        print("[weights] already exists:", WEIGHTS_PATH, flush=True)
        return True

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    tmp_path = WEIGHTS_PATH + ".download"
    print("[weights] downloading U2Net enhanced weights (~219MB, 3 parts)...",
          flush=True)

    received = 0
    last_pct = -10
    try:
        with open(tmp_path, "wb") as out:
            for name in U2NET_PARTS:
                print("[weights] part:", name, flush=True)
                with urllib.request.urlopen(U2NET_PARTS_BASE + name,
                                            timeout=180) as resp:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        received += len(chunk)
                        pct = min(100, received * 100 // U2NET_BYTES)
                        if pct >= last_pct + 10:
                            last_pct = pct
                            print("[weights] %d%%" % pct, flush=True)
    except Exception as exc:
        print("[weights] download failed:", exc, flush=True)
        print("[weights] 请确认仓库中的分片文件存在："
              "U2Net_/saved_models/parts/", flush=True)
        return False

    if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) != U2NET_BYTES:
        size = os.path.getsize(tmp_path) if os.path.isfile(tmp_path) else 0
        print("[weights] size mismatch: got %d, expected %d" % (size, U2NET_BYTES),
              flush=True)
        return False
    if _md5(tmp_path) != U2NET_MD5:
        print("[weights] MD5 mismatch, file corrupted", flush=True)
        return False

    os.replace(tmp_path, WEIGHTS_PATH)
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

    if weights_ready():
        print("u2net weights: OK", flush=True)
    else:
        print("u2net weights: MISSING or MISMATCHED "
              "(run: python colab_setup.py --download-weights)", flush=True)
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
