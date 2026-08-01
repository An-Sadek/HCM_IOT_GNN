import argparse
import csv
import gc
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from data import SEHTGNNDataset, collate_sehtgnn
from model import NodePredictor as SENodePredictor, SEHTGNN
from htgnn_model import HTGNN, NodePredictor as HTGNNNodePredictor


torch.manual_seed(42)


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_rmse",
    "train_r2",
    "train_mape",
    "val_loss",
    "val_rmse",
    "val_r2",
    "val_mape",
    "best_val_r2",
    "is_best",
]

NODE_TYPE_DESCRIPTIONS = {
    "segment": (
        "A road segment contains detailed information about its start node, "
        "end node, length, and the street to which it belongs."
    ),
    "node": (
        "A node is a point on Earth specified by longitude and latitude. "
        "It connects road segments together."
    ),
    "way": (
        "A street is formed by multiple road segments. It contains the street "
        "name, road level, road type, and maximum velocity under free-flow traffic."
    ),
}


def make_llm_feature(ntypes, model_name, dim=4096, device="cpu"):
    """Encode the author's descriptions into one semantic vector per node type."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "LLM features require transformers. Install it with `pip install transformers`."
        ) from exc

    unknown_types = set(ntypes) - NODE_TYPE_DESCRIPTIONS.keys()
    if unknown_types:
        raise ValueError(f"Missing LLM descriptions for node types: {sorted(unknown_types)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    prompts = [
        "Represent this urban road-network entity for graph learning: "
        + NODE_TYPE_DESCRIPTIONS[ntype]
        for ntype in ntypes
    ]
    encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    if embeddings.shape[1] != dim:
        raise ValueError(
            f"{model_name} produces {embeddings.shape[1]}-D embeddings, but "
            f"LLM4init expects {dim}. Choose a model with hidden size {dim}."
        )

    # LLM4init applies log(dot(source, destination)); make every component
    # positive so that all relation scores remain in the logarithm's domain.
    embeddings = embeddings - embeddings.amin(dim=1, keepdim=True)
    embeddings = F.normalize(embeddings + 1e-6, p=2, dim=1).cpu()
    llm_feature = {ntype: embeddings[i] for i, ntype in enumerate(ntypes)}

    # The language model is only needed to create these one-time CPU features.
    # Release it before constructing/training the GNN so it does not retain RAM/VRAM.
    del model, tokenizer, encoded, hidden, mask, embeddings
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

    return llm_feature


def update_regression_stats(stats, target, prediction, mape_epsilon):
    target = target.to(torch.float64)
    error = prediction.to(torch.float64) - target
    stats[0] += error.square().sum()
    stats[1] += (error.abs() / target.abs().clamp_min(mape_epsilon)).sum()
    stats[2] += target.sum()
    stats[3] += target.square().sum()
    stats[4] += target.numel()


def regression_metrics(stats):
    (
        squared_error,
        absolute_percentage_error,
        target_sum,
        target_squared_sum,
        count,
    ) = stats.tolist()
    count = max(count, 1.0)
    rmse = (squared_error / count) ** 0.5
    total_sum_of_squares = target_squared_sum - target_sum ** 2 / count
    r2 = (
        1.0 - squared_error / total_sum_of_squares
        if total_sum_of_squares > 0
        else float("nan")
    )
    mape = 100.0 * absolute_percentage_error / count
    return rmse, r2, mape


def chronological_split(dataset, train_ratio, val_ratio, split_gap):
    """Split sliding-window samples in time order without boundary overlap."""
    num_samples = len(dataset)
    usable_samples = num_samples - 2 * split_gap
    if usable_samples < 3:
        raise ValueError(
            "Not enough samples for chronological train/validation/test split: "
            f"samples={num_samples}, split_gap={split_gap}"
        )

    train_size = int(usable_samples * train_ratio)
    val_size = int(usable_samples * val_ratio)
    test_size = usable_samples - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            "Chronological split produced an empty partition: "
            f"train={train_size}, val={val_size}, test={test_size}"
        )

    train_start = 0
    train_end = train_start + train_size
    val_start = train_end + split_gap
    val_end = val_start + val_size
    test_start = val_end + split_gap
    test_end = test_start + test_size

    return (
        Subset(dataset, range(train_start, train_end)),
        Subset(dataset, range(val_start, val_end)),
        Subset(dataset, range(test_start, test_end)),
    )


def parse_args(default_architecture="sehtgnn"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("sehtgnn", "htgnn"),
        default=default_architecture,
        help="Model architecture. The dedicated HTGNN launcher defaults to htgnn.",
    )
    parser.add_argument("--preprocess-root", default="data/preprocess")
    parser.add_argument("--dynamic-path", default="data/preprocess/dynamic_features.npy")
    parser.add_argument(
        "--target-path",
        default="data/preprocess/dynamic_velocity.npy",
        help="Continuous velocity targets with shape (time, num_segments)",
    )
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Optional: stop after this many consecutive epochs without a meaningful "
            "validation R2 improvement. Disabled by default (0), so training runs "
            "for all --epochs while still saving the best validation-R2 checkpoint."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation R2 increase counted as an improvement.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mape-epsilon", type=float, default=1e-8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--velocity-channel",
        type=int,
        default=0,
        help="Channel in dynamic_features.npy containing raw velocity.",
    )
    parser.add_argument(
        "--split-gap",
        type=int,
        default=None,
        help=(
            "Number of sliding-window sample starts omitted at each split boundary. "
            "Defaults to window_size + horizon - 1 so partitions share no timestamps."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Checkpoint to resume training from. Restores model weights and, when "
            "available, optimizer/epoch/best validation state. --epochs is the number "
            "of additional epochs to train."
        ),
    )
    if default_architecture == "htgnn":
        default_checkpoint = "result/htgnn_best.pt"
        default_history = "result/htgnn_train_history.csv"
        default_test = "result/htgnn_test.csv"
    else:
        # Preserve the original SE-HTGNN output paths.
        default_checkpoint = "result/sehtgnn_best.pt"
        default_history = "result/train_history.csv"
        default_test = "result/test.csv"
    parser.add_argument("--checkpoint", default=default_checkpoint)
    parser.add_argument("--history-csv", default=default_history)
    parser.add_argument("--test-csv", default=default_test)
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model used to encode node-type descriptions (hidden size 4096)",
    )
    parser.add_argument(
        "--llm-device",
        default="cuda",
        help="Device used once to create LLM embeddings, e.g. cpu, cuda, or cuda:0",
    )
    return parser.parse_args()


def main(default_architecture="sehtgnn"):
    args = parse_args(default_architecture=default_architecture)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be >= 0")
    if args.mape_epsilon <= 0:
        raise ValueError("--mape-epsilon must be > 0")
    split_gap = (
        args.window_size + args.horizon - 1
        if args.split_gap is None
        else args.split_gap
    )
    if split_gap < 0:
        raise ValueError("--split-gap must be >= 0")
    split_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if min(args.train_ratio, args.val_ratio, args.test_ratio) <= 0:
        raise ValueError("Train, validation, and test ratios must all be positive")
    if abs(split_ratio - 1.0) > 1e-8:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1")

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs with the NCCL backend")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(args.device)
        rank = 0
        world_size = 1
    is_main = rank == 0
    if is_main:
        print(
            f"training_mode={'DDP' if distributed else 'single-process'} "
            f"world_size={world_size} batch_size_per_gpu={args.batch_size}"
        )

    dataset = SEHTGNNDataset(
        preprocess_root=args.preprocess_root,
        dynamic_path=args.dynamic_path,
        target_path=args.target_path,
        window_size=args.window_size,
        horizon=args.horizon,
    )

    train_ds, val_ds, test_ds = chronological_split(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_gap=split_gap,
    )
    train_size, val_size, test_size = map(len, (train_ds, val_ds, test_ds))
    train_timestamp_start = train_ds.indices[0]
    train_timestamp_end = (
        train_ds.indices[-1] + args.window_size + args.horizon
    )
    velocity_mean, velocity_scale = dataset.fit_input_velocity_scaler(
        timestamp_start=train_timestamp_start,
        timestamp_end=train_timestamp_end,
        velocity_channel=args.velocity_channel,
    )
    if is_main:
        train_first, train_last = train_ds.indices[0], train_ds.indices[-1]
        val_first, val_last = val_ds.indices[0], val_ds.indices[-1]
        test_first, test_last = test_ds.indices[0], test_ds.indices[-1]
        sample_span = args.window_size + args.horizon
        print(
            "chronological_split "
            f"gap={split_gap} sample_span={sample_span} "
            f"train={train_size}[{train_first}:{train_last}] "
            f"val={val_size}[{val_first}:{val_last}] "
            f"test={test_size}[{test_first}:{test_last}]"
        )
        print(
            "input_velocity_scaler "
            f"fit_timestamps=[{train_timestamp_start}:{train_timestamp_end - 1}] "
            f"channel={args.velocity_channel} per_segment=True "
            f"mean_range=[{velocity_mean.min():.4f}, {velocity_mean.max():.4f}] "
            f"scale_range=[{velocity_scale.min():.4f}, {velocity_scale.max():.4f}]"
        )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if distributed else None
    test_sampler = DistributedSampler(
        test_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_sehtgnn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_sehtgnn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_sehtgnn,
    )

    resume_path = Path(args.model) if args.model else None
    resume_checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=True,
        )
        checkpoint_architecture = resume_checkpoint.get("architecture")
        if (
            checkpoint_architecture is not None
            and checkpoint_architecture != args.architecture
        ):
            raise ValueError(
                f"Checkpoint architecture is {checkpoint_architecture!r}, but "
                f"the requested architecture is {args.architecture!r}"
            )

    sample_graph, _ = dataset[0]
    llm_feature = None
    if args.architecture == "sehtgnn":
        if resume_checkpoint is not None and "llm_feature" in resume_checkpoint:
            llm_feature = resume_checkpoint["llm_feature"]
            if is_main:
                print(f"loaded_llm_feature_from={resume_path}")
        else:
            if is_main and resume_checkpoint is not None:
                print(
                    "resume checkpoint has no llm_feature; generating it once for "
                    "backward compatibility"
                )
            if is_main:
                llm_feature = make_llm_feature(
                    sample_graph.ntypes,
                    model_name=args.llm_model,
                    device=args.llm_device,
                )
            if distributed:
                llm_feature_container = [llm_feature]
                dist.broadcast_object_list(llm_feature_container, src=0, device=device)
                llm_feature = llm_feature_container[0]

    encoder_class = HTGNN if args.architecture == "htgnn" else SEHTGNN
    predictor_class = (
        HTGNNNodePredictor if args.architecture == "htgnn" else SENodePredictor
    )
    encoder = encoder_class(
        graph=sample_graph,
        n_inp=args.hidden_dim,
        n_hid=args.hidden_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        time_window=args.window_size,
        norm=True,
        device=device,
        dropout=args.dropout,
        LLM_feature=llm_feature,
        inp_list=dataset.inp_list,
    ).to(device)
    predictor = predictor_class(args.hidden_dim, args.horizon).to(device)

    if distributed:
        # The current model contains parameters that are not used by every forward path.
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
        predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.MSELoss()

    best_val_r2 = float("-inf")
    epochs_without_improvement = 0
    start_epoch = 0
    saved_best_this_run = False
    if resume_checkpoint is not None:
        encoder_to_load = encoder.module if distributed else encoder
        predictor_to_load = predictor.module if distributed else predictor
        encoder_to_load.load_state_dict(resume_checkpoint["encoder"])
        predictor_to_load.load_state_dict(resume_checkpoint["predictor"])
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_val_r2 = float(resume_checkpoint.get("best_val_r2", float("-inf")))
        if is_main:
            print(
                f"resumed_from={resume_path} start_epoch={start_epoch + 1} "
                f"best_val_r2={best_val_r2:.4f}"
            )

    checkpoint_path = Path(args.checkpoint)
    if is_main:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_csv)
    if is_main:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        if resume_path is None or not history_path.exists():
            with history_path.open("w", newline="") as history_file:
                writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
                writer.writeheader()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        encoder.train()
        predictor.train()
        train_loss_sum = 0.0
        train_count = 0
        train_metric_stats = torch.zeros(5, dtype=torch.float64, device=device)
        optimizer.zero_grad()

        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch:03d} train",
            leave=False,
            disable=not is_main,
        )
        for step, (graph, y) in enumerate(train_bar, start=1):
            graph = graph.to(device)
            y = y.to(device)

            segment_emb = encoder(graph, predict_type="segment")
            predictions = predictor(segment_emb).view(-1, args.horizon)
            loss = criterion(predictions, y)
            (loss / args.grad_accum_steps).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            target_count = y.numel()
            train_loss_sum += loss.item() * target_count
            train_count += target_count
            update_regression_stats(
                train_metric_stats, y, predictions.detach(), args.mape_epsilon
            )
            accum_step = (step - 1) % args.grad_accum_steps + 1
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                rmse=f"{(train_metric_stats[0].item() / max(train_count, 1)) ** 0.5:.4f}",
                accum=f"{accum_step}/{args.grad_accum_steps}",
            )

        train_stats = torch.tensor(
            [train_loss_sum, train_count],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_metric_stats, op=dist.ReduceOp.SUM)
        train_loss_sum, train_count = train_stats.tolist()
        train_loss = train_loss_sum / max(train_count, 1)
        train_rmse, train_r2, train_mape = regression_metrics(train_metric_stats)

        encoder.eval()
        predictor.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_metric_stats = torch.zeros(5, dtype=torch.float64, device=device)
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"epoch {epoch:03d} val",
                leave=False,
                disable=not is_main,
            )
            for graph, y in val_bar:
                graph = graph.to(device)
                y = y.to(device)
                predictions = predictor(encoder(graph, predict_type="segment")).view(
                    -1, args.horizon
                )
                loss = criterion(predictions, y)
                target_count = y.numel()
                val_loss_sum += loss.item() * target_count
                val_count += target_count
                update_regression_stats(
                    val_metric_stats, y, predictions, args.mape_epsilon
                )
                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    rmse=f"{(val_metric_stats[0].item() / max(val_count, 1)) ** 0.5:.4f}",
                )

        val_stats = torch.tensor(
            [val_loss_sum, val_count],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_metric_stats, op=dist.ReduceOp.SUM)
        val_loss_sum, val_count = val_stats.tolist()
        val_loss = val_loss_sum / max(val_count, 1)
        val_rmse, val_r2, val_mape = regression_metrics(val_metric_stats)
        if is_main:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_rmse={train_rmse:.4f} "
                f"train_r2={train_r2:.4f} train_mape={train_mape:.2f}% "
                f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f} "
                f"val_r2={val_r2:.4f} val_mape={val_mape:.2f}%"
            )

        is_best = val_r2 > best_val_r2 + args.early_stopping_min_delta
        if is_best:
            best_val_r2 = val_r2
            epochs_without_improvement = 0
            saved_best_this_run = True
        else:
            epochs_without_improvement += 1
        if is_main and is_best:
            checkpoint_data = {
                    "encoder": (encoder.module if distributed else encoder).state_dict(),
                    "predictor": (predictor.module if distributed else predictor).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_r2": best_val_r2,
                    "architecture": args.architecture,
                    "args": vars(args),
                    "inp_list": dataset.inp_list,
                }
            if llm_feature is not None:
                checkpoint_data["llm_feature"] = {
                    key: feature.detach().cpu()
                    for key, feature in llm_feature.items()
                }
            torch.save(checkpoint_data, checkpoint_path)
        if is_main:
            with history_path.open("a", newline="") as history_file:
                writer = csv.DictWriter(
                    history_file,
                    fieldnames=HISTORY_FIELDS,
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_rmse": train_rmse,
                        "train_r2": train_r2,
                        "train_mape": train_mape,
                        "val_loss": val_loss,
                        "val_rmse": val_rmse,
                        "val_r2": val_r2,
                        "val_mape": val_mape,
                        "best_val_r2": best_val_r2,
                        "is_best": int(is_best),
                    }
                )

        should_stop = (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        )
        if should_stop:
            if is_main:
                print(
                    f"early_stopping epoch={epoch:03d} "
                    f"patience={args.early_stopping_patience} "
                    f"best_val_r2={best_val_r2:.4f}"
                )
            break

    if distributed:
        dist.barrier()

    best_model_path = checkpoint_path if saved_best_this_run else resume_path
    if best_model_path is None:
        raise FileNotFoundError(
            f"No checkpoint was saved to {checkpoint_path}; cannot run test evaluation"
        )
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    (encoder.module if distributed else encoder).load_state_dict(checkpoint["encoder"])
    (predictor.module if distributed else predictor).load_state_dict(checkpoint["predictor"])
    encoder.eval()
    predictor.eval()

    test_loss_sum = 0.0
    test_count = 0
    test_metric_stats = torch.zeros(5, dtype=torch.float64, device=device)
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="test", leave=False, disable=not is_main)
        for graph, y in test_bar:
            graph = graph.to(device)
            y = y.to(device)
            prediction = predictor(encoder(graph, predict_type="segment")).view(
                -1, args.horizon
            )
            loss = criterion(prediction, y)
            target_count = y.numel()
            test_loss_sum += loss.item() * target_count
            test_count += target_count
            update_regression_stats(
                test_metric_stats, y, prediction, args.mape_epsilon
            )

    test_stats = torch.tensor(
        [test_loss_sum, test_count],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(test_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_metric_stats, op=dist.ReduceOp.SUM)
    test_loss_sum, test_count = test_stats.tolist()
    test_loss = test_loss_sum / max(test_count, 1)
    test_rmse, test_r2, test_mape = regression_metrics(test_metric_stats)

    if is_main:
        test_path = Path(args.test_csv)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with test_path.open("w", newline="") as test_file:
            writer = csv.DictWriter(
                test_file,
                fieldnames=["test_size", "test_loss", "test_rmse", "test_r2", "test_mape"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "test_size": test_size,
                    "test_loss": test_loss,
                    "test_rmse": test_rmse,
                    "test_r2": test_r2,
                    "test_mape": test_mape,
                }
            )
        print(f"best_val_r2={best_val_r2:.4f}")
        print(
            f"test_loss={test_loss:.4f} test_rmse={test_rmse:.4f} "
            f"test_r2={test_r2:.4f} test_mape={test_mape:.2f}% test_csv={test_path}"
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
