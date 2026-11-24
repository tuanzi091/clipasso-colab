#!/usr/bin/env python
"""批量运行 target_images/ 全部图片：实时输出进度 + 笔触/风格反馈，最后拼图并打包。

用法（Colab）：
    cd /content/clipasso-colab
    python -u batch_run.py                     # 全量，默认 2000 iter
    python -u batch_run.py --iter 800 --only camel,cat   # 小规模试验
    python -u batch_run.py --pack_only         # 只重新拼图+打包
"""
import os
import re
import sys
import glob
import time
import json
import argparse
import subprocess
import zipfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(REPO_ROOT, "target_images")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

# tqdm 进度行：  40%|████      | 800/2000 [02:30<03:45,  6.5it/s]
TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[([0-9:]+)<")
KEY_RE = re.compile(
    r"\[Step|\[Complexity\]|\[Dynamic|\[Drawing Style\]|Drawing Style|"
    r"Width Range|Opacity Range|Formula|Description|Running:|Using external mask|"
    r"Stage|Saved|Successfully|Error|Traceback|Exception|No module"
)


def collect_images(input_dir, only=None):
    files = sorted(
        f for f in glob.glob(os.path.join(input_dir, "*"))
        if os.path.isfile(f) and f.lower().endswith(IMG_EXTS)
    )
    if only:
        white = {n.strip() for n in only.split(",") if n.strip()}
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in white]
    return files


def run_one_live(img_path, out_dir, num_paths, num_iter, use_gpu, idx, total):
    """跑单张，边跑边打印关键信息和百分比进度。"""
    cmd = [
        sys.executable, "-u", "process_image.py",
        img_path,
        "--output_dir", out_dir,
        "--num_paths", str(num_paths),
        "--num_iter", str(num_iter),
        "--save_interval", "50",
        "--use_gpu", str(int(use_gpu)),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(REPO_ROOT, "diffvg")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*66}")
    print(f"[{idx}/{total}] {name}")
    print(f"{'='*66}", flush=True)

    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=0,
    )

    buf = ""
    last_pct = -10
    t0 = time.time()

    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(r"[\r\n]+", buf)
        buf = parts.pop()
        for line in parts:
            line = line.strip()
            if not line:
                continue

            m = TQDM_RE.search(line)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                pct = cur * 100 // max(tot, 1)
                if pct >= last_pct + 10:
                    last_pct = pct
                    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                    print(f"    进度 [{bar}] {pct:3d}%  {cur}/{tot}  "
                          f"用时 {time.time()-t0:5.0f}s", flush=True)
                continue

            if KEY_RE.search(line):
                print(f"    {line}", flush=True)

    proc.wait()
    return proc.returncode, time.time() - t0


def find_result(out_root, name, pattern):
    hits = glob.glob(os.path.join(out_root, name, "**", pattern), recursive=True)
    return sorted(hits)[-1] if hits else None


def build_montage(out_root, cols=4, cell=340):
    from PIL import Image, ImageDraw

    items = [(d, find_result(out_root, d, "best_iter.jpg"))
             for d in sorted(os.listdir(out_root))
             if os.path.isdir(os.path.join(out_root, d))]
    items = [(n, p) for n, p in items if p]
    if not items:
        print("[拼图] 没有找到 best_iter.jpg，跳过")
        return None

    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, path) in enumerate(items):
        r, c = divmod(i, cols)
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            continue
        im.thumbnail((cell - 20, cell - 46))
        canvas.paste(im, (c * cell + (cell - im.width) // 2, r * cell + 30))
        draw.text((c * cell + 10, r * cell + 8), name, fill="black")

    dest = os.path.join(out_root, "preview_montage.jpg")
    canvas.save(dest, quality=90)
    print(f"[拼图] {len(items)} 张 -> {dest}")
    return dest


def build_zip(out_root, dest=None):
    wanted = ("original.png", "foreground.png", "mask.png",
              "best_iter.jpg", "best_iter.svg", "final_svg.svg", "best_adaptive.jpg")
    dest = dest or os.path.join(out_root, "clipasso_batch.zip")
    n = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in glob.glob(os.path.join(out_root, "**", "*"), recursive=True):
            if not os.path.isfile(f):
                continue
            base = os.path.basename(f)
            in_root = os.path.dirname(os.path.abspath(f)) == os.path.abspath(out_root)
            if base in wanted or (in_root and base in ("preview_montage.jpg", "batch_summary.json")):
                zf.write(f, os.path.relpath(f, out_root).replace("\\", "/"))
                n += 1
    print(f"[打包] {n} 个文件 -> {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
    return dest


def do_pack(out_root):
    build_montage(out_root)
    return build_zip(out_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=INPUT_DIR)
    ap.add_argument("--output_root", default=os.path.join(REPO_ROOT, "results_batch"))
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--paths", type=int, default=24)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip_done", type=int, default=1)
    ap.add_argument("--pack_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    if args.pack_only:
        do_pack(args.output_root)
        return

    images = collect_images(args.input_dir, args.only)
    assert images, f"在 {args.input_dir} 下没找到图片"

    print(f"{'='*66}")
    print(f"共 {len(images)} 张图 | iter={args.iter} | paths={args.paths} | gpu={args.gpu}")
    print(f"图片: {', '.join(os.path.splitext(os.path.basename(p))[0] for p in images)}")
    print(f"{'='*66}", flush=True)

    summary = []
    done_times = []
    t_all = time.time()

    for i, img in enumerate(images, 1):
        name = os.path.splitext(os.path.basename(img))[0]
        out_dir = os.path.join(args.output_root, name)
        marker = os.path.join(out_dir, ".done")

        if args.skip_done and os.path.exists(marker):
            print(f"\n[{i}/{len(images)}] {name} 已完成，跳过")
            summary.append({"name": name, "status": "SKIP", "elapsed_s": 0})
            continue

        os.makedirs(out_dir, exist_ok=True)
        rc, elapsed = run_one_live(img, out_dir, args.paths, args.iter, args.gpu, i, len(images))

        if rc == 0:
            with open(marker, "w") as f:
                f.write(f"{args.iter} iters, {elapsed:.1f}s, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            summary.append({"name": name, "status": "OK", "elapsed_s": round(elapsed, 1)})
            done_times.append(elapsed)
            eta = (sum(done_times) / len(done_times)) * (len(images) - i) / 60
            print(f"    ✓ {name} 完成 {elapsed/60:.1f} min | 预计剩余 {eta:.0f} min", flush=True)
        else:
            summary.append({"name": name, "status": f"FAIL(rc={rc})", "elapsed_s": round(elapsed, 1)})
            print(f"    ✗ {name} 失败 rc={rc}", flush=True)

    total = time.time() - t_all
    print(f"\n{'='*66}")
    print(f"总用时 {total/60:.1f} min | 成功 "
          f"{sum(1 for s in summary if s['status']=='OK')}/{len(summary)}")
    print(f"{'name':<18}{'status':<14}{'min':>8}")
    print("-" * 40)
    for s in summary:
        print(f"{s['name']:<18}{s['status']:<14}{s['elapsed_s']/60:>8.1f}")
    print(f"{'='*66}", flush=True)

    json.dump(
        {"total_min": round(total/60, 1),
         "params": {"iter": args.iter, "paths": args.paths, "gpu": args.gpu},
         "results": summary},
        open(os.path.join(args.output_root, "batch_summary.json"), "w"),
        indent=2, ensure_ascii=False,
    )

    do_pack(args.output_root)


if __name__ == "__main__":
    main()
