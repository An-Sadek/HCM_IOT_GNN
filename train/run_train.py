import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data import SEHTGNNDataset, collate_sehtgnn
from model import NodePredictor, SEHTGNN


def make_llm_feature(ntypes, dim=4096):
    # Positive vectors keep the official LLM4init log(inner_product) well-defined.
    return {ntype: torch.ones(dim) for ntype in ntypes}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess-root", default="data/preprocess")
    parser.add_argument("--dynamic-path", default="data/preprocess/dynamic_features.npy")
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default="data/preprocess/sehtgnn_best.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = SEHTGNNDataset(
        preprocess_root=args.preprocess_root,
        dynamic_path=args.dynamic_path,
        window_size=args.window_size,
        horizon=args.horizon,
        target_channel=0,
    )

    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sehtgnn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sehtgnn,
    )

    sample_graph, _ = dataset[0]
    llm_feature = make_llm_feature(sample_graph.ntypes)

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
    predictor = NodePredictor(args.hidden_dim, 6 * args.horizon).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        train_loss = 0.0

        for graph, y in train_loader:
            graph = graph.to(device)
            y = y.to(device).long()

            optimizer.zero_grad()
            segment_emb = encoder(graph, predict_type="segment")
            logits = predictor(segment_emb).view(-1, args.horizon, 6)
            loss = criterion(logits.reshape(-1, 6), y.reshape(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        encoder.eval()
        predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph, y in val_loader:
                graph = graph.to(device)
                y = y.to(device).long()
                logits = predictor(encoder(graph, predict_type="segment")).view(
                    -1,
                    args.horizon,
                    6,
                )
                val_loss += criterion(logits.reshape(-1, 6), y.reshape(-1)).item()

        val_loss /= max(len(val_loader), 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "args": vars(args),
                    "inp_list": dataset.inp_list,
                },
                checkpoint_path,
            )

    print(f"best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
