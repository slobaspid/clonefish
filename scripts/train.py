"""Train a (clock-aware) Chessformer on shard(s).

Usage (Maia-recipe ~19.5M clock-aware model on a rented GPU):
    PYTHONPATH=. python scripts/train.py "data/chesscom_shards/*.npz" \
        --mode full --dim-vit 512 --num-blocks 8 \
        --lr 5e-5 --weight-decay 1e-6 --grad-clip 3.5 --warmup-steps 1000 \
        --batch-size 512 --max-steps 200000 --amp --stream --device cuda \
        --out checkpoints/full --resume checkpoints/full/last.pt
"""
import argparse
import glob
from sahformer.model.config import ModelConfig
from sahformer.training.loop import TrainConfig, train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", help="glob for .npz shard(s)")
    ap.add_argument("--mode", default="full",
                    choices=["baseline", "film_only", "gab_only", "full"])
    # model size (Maia rungs: 256/8 ~5M, 512/8 ~19.5M, 1024/8 ~72M)
    ap.add_argument("--dim-vit", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=8)
    # optimization (defaults below match Maia-3's published recipe)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=3.5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--stream", action="store_true", help="stream shards (huge datasets)")
    ap.add_argument("--resume", default="", help="path to a checkpoint (last.pt) to continue from")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.shards))
    if not paths:
        raise SystemExit(f"no shards matched: {args.shards}")

    model_cfg = ModelConfig(dim_vit=args.dim_vit, num_blocks=args.num_blocks)
    cfg = TrainConfig(mode=args.mode, max_steps=args.max_steps, batch_size=args.batch_size,
                      lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                      warmup_steps=args.warmup_steps, out_dir=args.out, amp=args.amp,
                      stream=args.stream, resume=args.resume, device=args.device,
                      log_every=args.log_every, ckpt_every=args.ckpt_every)
    res = train(cfg, paths, model_cfg=model_cfg)
    print(f"done. best_total={res['best']:.4f} steps={len(res['history'])}")

if __name__ == "__main__":
    main()
