import argparse
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from data import NUM_CLASSES, SEHTGNNDataset, collate_sehtgnn
from model import NodePredictor, SEHTGNN


torch.manual_seed(42)


HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "train_accuracy",
    "train_f1_macro",
    "val_loss",
    "val_accuracy",
    "val_f1_macro",
    "best_val_f1_macro",
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
    return {ntype: embeddings[i] for i, ntype in enumerate(ntypes)}


def update_confusion_matrix(confusion, target, prediction):
    if target.ndim == prediction.ndim and target.shape[-1] == NUM_CLASSES:
        target = target.argmax(dim=-1)
    if prediction.ndim > target.ndim:
        prediction = prediction.argmax(dim=-1)
    indices = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    confusion += torch.bincount(
        indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def classification_metrics(confusion):
    confusion = confusion.to(torch.float64)
    true_positive = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    accuracy = true_positive.sum() / confusion.sum().clamp_min(1)
    f1_per_class = 2 * true_positive / (support + predicted).clamp_min(1)
    present_classes = support > 0
    macro_f1 = f1_per_class[present_classes].mean() if present_classes.any() else 0.0
    return accuracy.item(), float(macro_f1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-root", default="data/preprocess")
    parser.add_argument("--dynamic-path", default="data/preprocess/dynamic_features.npy")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="result/sehtgnn_best.pt")
    parser.add_argument("--history-csv", default="result/train_history.csv")
    parser.add_argument("--test-csv", default="result/test.csv")
    parser.add_argument(
        "--llm-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model used to encode node-type descriptions (hidden size 4096)",
    )
    parser.add_argument(
        "--llm-device",
        default="cpu",
        help="Device used once to create LLM embeddings, e.g. cpu, cuda, or cuda:0",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
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
        window_size=args.window_size,
        horizon=args.horizon,
        target_channel=0,
        num_classes=NUM_CLASSES,
    )

    train_size = int(len(dataset) * args.train_ratio)
    val_size = int(len(dataset) * args.val_ratio)
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
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

    sample_graph, _ = dataset[0]
    llm_feature = make_llm_feature(
        sample_graph.ntypes,
        model_name=args.llm_model,
        device=args.llm_device,
    )

    encoder = SEHTGNN(
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
    predictor = NodePredictor(args.hidden_dim, NUM_CLASSES * args.horizon).to(device)

    if distributed:
        # The current model contains parameters that are not used by every forward path.
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
        predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    checkpoint_path = Path(args.checkpoint)
    if is_main:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_csv)
    if is_main:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        encoder.train()
        predictor.train()
        train_loss_sum = 0.0
        train_count = 0
        train_confusion = torch.zeros(
            NUM_CLASSES,
            NUM_CLASSES,
            dtype=torch.int64,
            device=device,
        )
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
            logits = predictor(segment_emb).view(-1, args.horizon, NUM_CLASSES)
            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                y.reshape(-1, NUM_CLASSES),
            )
            (loss / args.grad_accum_steps).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            pred_one_hot = F.one_hot(
                logits.detach().argmax(dim=-1),
                num_classes=NUM_CLASSES,
            )
            target_count = y.shape[:-1].numel()
            train_loss_sum += loss.item() * target_count
            train_count += target_count
            update_confusion_matrix(train_confusion, y, pred_one_hot)
            accum_step = (step - 1) % args.grad_accum_steps + 1
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{train_confusion.diag().sum().item() / max(train_count, 1):.4f}",
                accum=f"{accum_step}/{args.grad_accum_steps}",
            )

        train_stats = torch.tensor(
            [train_loss_sum, train_count],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_confusion, op=dist.ReduceOp.SUM)
        train_loss_sum, train_count = train_stats.tolist()
        train_loss = train_loss_sum / max(train_count, 1)
        train_accuracy, train_f1 = classification_metrics(train_confusion)

        encoder.eval()
        predictor.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_confusion = torch.zeros(
            NUM_CLASSES,
            NUM_CLASSES,
            dtype=torch.int64,
            device=device,
        )
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
                logits = predictor(encoder(graph, predict_type="segment")).view(
                    -1,
                    args.horizon,
                    NUM_CLASSES,
                )
                loss = criterion(
                    logits.reshape(-1, NUM_CLASSES),
                    y.reshape(-1, NUM_CLASSES),
                )
                pred_one_hot = F.one_hot(
                    logits.argmax(dim=-1),
                    num_classes=NUM_CLASSES,
                )
                target_count = y.shape[:-1].numel()
                val_loss_sum += loss.item() * target_count
                val_count += target_count
                update_confusion_matrix(val_confusion, y, pred_one_hot)
                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    accuracy=f"{val_confusion.diag().sum().item() / max(val_count, 1):.4f}",
                )

        val_stats = torch.tensor(
            [val_loss_sum, val_count],
            dtype=torch.float64,
            device=device,
        )
        if distributed:
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_confusion, op=dist.ReduceOp.SUM)
        val_loss_sum, val_count = val_stats.tolist()
        val_loss = val_loss_sum / max(val_count, 1)
        val_accuracy, val_f1 = classification_metrics(val_confusion)
        if is_main:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} "
                f"train_f1={train_f1:.4f} val_loss={val_loss:.4f} "
                f"val_accuracy={val_accuracy:.4f} val_f1={val_f1:.4f}"
            )

        is_best = val_f1 > best_val_f1
        if is_main and is_best:
            best_val_f1 = val_f1
            torch.save(
                {
                    "encoder": (encoder.module if distributed else encoder).state_dict(),
                    "predictor": (predictor.module if distributed else predictor).state_dict(),
                    "args": vars(args),
                    "inp_list": dataset.inp_list,
                },
                checkpoint_path,
            )
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
                        "train_accuracy": train_accuracy,
                        "train_f1_macro": train_f1,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "val_f1_macro": val_f1,
                        "best_val_f1_macro": best_val_f1,
                        "is_best": int(is_best),
                    }
                )

    if distributed:
        dist.barrier()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    (encoder.module if distributed else encoder).load_state_dict(checkpoint["encoder"])
    (predictor.module if distributed else predictor).load_state_dict(checkpoint["predictor"])
    encoder.eval()
    predictor.eval()

    test_loss_sum = 0.0
    test_count = 0
    test_confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.int64,
        device=device,
    )
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="test", leave=False, disable=not is_main)
        for graph, y in test_bar:
            graph = graph.to(device)
            y = y.to(device)
            logits = predictor(encoder(graph, predict_type="segment")).view(
                -1,
                args.horizon,
                NUM_CLASSES,
            )
            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                y.reshape(-1, NUM_CLASSES),
            )
            prediction = F.one_hot(logits.argmax(dim=-1), num_classes=NUM_CLASSES)
            target_count = y.shape[:-1].numel()
            test_loss_sum += loss.item() * target_count
            test_count += target_count
            update_confusion_matrix(test_confusion, y, prediction)

    test_stats = torch.tensor(
        [test_loss_sum, test_count],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(test_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_confusion, op=dist.ReduceOp.SUM)
    test_loss_sum, test_count = test_stats.tolist()
    test_loss = test_loss_sum / max(test_count, 1)
    test_accuracy, test_f1 = classification_metrics(test_confusion)

    if is_main:
        test_path = Path(args.test_csv)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with test_path.open("w", newline="") as test_file:
            writer = csv.DictWriter(
                test_file,
                fieldnames=["test_size", "test_loss", "test_accuracy", "test_f1_macro"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "test_size": test_size,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "test_f1_macro": test_f1,
                }
            )
        print(f"best_val_f1_macro={best_val_f1:.4f}")
        print(
            f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f} "
            f"test_f1={test_f1:.4f} test_csv={test_path}"
        )
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
