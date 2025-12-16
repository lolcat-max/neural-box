import os
import time
from collections import Counter
from tqdm import tqdm

import serial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# Config
# -------------------------
CKPT_PATH = "rnn_alt_fgsm_model.pth"
DATASET_PT_PATH = "analog_xaa_dataset.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocab source
XAA_PATH = "xaa"
VOCAB_MAX = 8000          # cap vocab to avoid gigantic embedding (set None for unlimited)
MIN_FREQ = 1

# Model/data hyperparams (saved into checkpoint too)
SEQ_LEN = 8
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1

# Training hyperparams
BATCH_SIZE = 512
LR = 1e-2
NUM_EPOCHS = 1

EPS_START = 0.10
EPS_MAX = 0.30
EPS_GROW_EVERY = 4
EPS_GROW_MULT = 1.15

ADV_EVERY = 4
EMB_CLAMP = 2.0
GRAD_CLIP_NORM = 1.0

# Serial
SERIAL_PORT = "COM3"      # <<< change
SERIAL_BAUD = 115200
SERIAL_TIMEOUT_S = 1.0
SERIAL_NUM_IDS = 300000
SERIAL_PROGRESS_EVERY = 10000

# Caching
CACHE_DATASET_PT = True

# -------------------------
# Vocab from xaa (word labels)
# -------------------------
def build_vocab_from_xaa(path, vocab_max=8000, min_freq=1):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()

    words = text.split()
    counts = Counter(words)

    # keep most common tokens (optionally)
    items = [(w, c) for (w, c) in counts.items() if c >= int(min_freq)]
    items.sort(key=lambda wc: (-wc[1], wc[0]))

    vocab = ["<unk>"]
    if vocab_max is None:
        vocab.extend([w for (w, _) in items if w != "<unk>"])
    else:
        for (w, _) in items:
            if w == "<unk>":
                continue
            vocab.append(w)
            if len(vocab) >= int(vocab_max):
                break

    word_to_ix = {w: i for i, w in enumerate(vocab)}
    ix_to_word = {i: w for w, i in word_to_ix.items()}
    return word_to_ix, ix_to_word

# -------------------------
# Serial -> ids tensor
# -------------------------
def serial_handshake_and_stream_ids(port, baud, vocab_size, num_ids,
                                   timeout_s=1.0, progress_every=10000):
    ser = serial.Serial(port, baud, timeout=timeout_s)

    # reset many Arduino boards by toggling DTR
    ser.dtr = False
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.dtr = True
    time.sleep(1.5)

    # wait for READY (repeats every 500ms in sketch)
    deadline = time.time() + 10.0
    ready = False
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ARDUINO:", line)
        if line == "READY":
            ready = True
            break
    if not ready:
        ser.close()
        raise RuntimeError("No READY from Arduino. Check port/baud, close Serial Monitor, confirm sketch flashed.")

    # ping
    ser.write(b"PING\n")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("ARDUINO:", line)
        if line == "PONG":
            break

    # configure + go
    ser.write(f"V{int(vocab_size)}\n".encode())
    ser.write(f"N{int(num_ids)}\n".encode())
    ser.write(b"GO\n")

    ids = torch.empty((int(num_ids),), dtype=torch.long)
    got = 0
    t0 = time.time()

    while got < num_ids:
        s = ser.readline().decode(errors="ignore").strip()
        if not s:
            continue

        if s in ("READY", "PONG", "START") or s.startswith("OK"):
            continue
        if s.startswith("ERR"):
            print("ARDUINO:", s)
            continue
        if s == "DONE":
            break

        try:
            v = int(s)
        except ValueError:
            print("ARDUINO(non-int):", repr(s))
            continue

        # keep in-range for embedding/classification
        ids[got] = v % int(vocab_size)
        got += 1

        if progress_every and got % int(progress_every) == 0:
            dt = max(1e-6, time.time() - t0)
            print(f"Serial ids: {got}/{num_ids} ({got/dt:.0f} lines/s)")

    ser.close()
    return ids[:got].contiguous()

def make_next_token_dataset_from_ids(ids_t, seq_len):
    L = int(ids_t.numel())
    if L <= int(seq_len):
        raise ValueError(f"Need more ids for seq_len={seq_len}: got {L}")

    # X: sliding windows; y: next token
    X_all = ids_t.unfold(0, int(seq_len), 1)         # (L-seq_len+1, seq_len)
    X = X_all[: L - int(seq_len)].contiguous()       # (L-seq_len, seq_len)
    y = ids_t[int(seq_len):].contiguous()            # (L-seq_len,)
    return TensorDataset(X, y)

# -------------------------
# Custom non-commutative op: (A @ B)
# -------------------------
class NonCommutativeMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B

    @staticmethod
    def backward(ctx, grad_out):
        A, B = ctx.saved_tensors
        grad_A = grad_out @ B.transpose(-1, -2)
        grad_B = A.transpose(-1, -2) @ grad_out
        return grad_A, grad_B

nc_matmul = NonCommutativeMatMul.apply

class NCLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = nc_matmul(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            y = y + self.bias
        return y

# -------------------------
# Model
# -------------------------
class RNNNextWord(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = NCLinear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        emb = self.embedding(x)
        out, h_next = self.rnn(emb, h)
        logits = self.fc_out(out[:, -1, :])
        return logits, h_next

    def forward_step(self, token_id, h=None):
        emb = self.embedding(token_id)
        out, h_next = self.rnn(emb, h)
        logits = self.fc_out(out[:, -1, :])
        return logits, h_next

# -------------------------
# FGSM on embeddings
# -------------------------
def fgsm_embeddings(model, x, y, criterion, epsilon, clamp_val=2.0):
    emb = model.embedding(x).detach().requires_grad_(True)
    out, _ = model.rnn(emb)
    logits = model.fc_out(out[:, -1, :])
    loss = criterion(logits, y)

    grad = torch.autograd.grad(loss, emb, retain_graph=False, create_graph=False)[0]
    emb_adv = emb + float(epsilon) * grad.sign()
    emb_adv = torch.clamp(emb_adv, -float(clamp_val), float(clamp_val)).detach()
    return emb_adv

# -------------------------
# Train (alternating)
# -------------------------
def train_epoch_alternating(model, optimizer, criterion, loader,
                            epsilon=0.15, adv_every=2, clamp_val=2.0):
    model.train()
    total_loss, batches = 0.0, 0

    progress_bar = tqdm(loader, desc=f"Epoch alt eps={epsilon:.2f}", leave=False)
    for step, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        do_adv = (step % int(adv_every)) == (int(adv_every) - 1)

        if not do_adv:
            logits, _ = model(x)
            loss = criterion(logits, y)
        else:
            emb_adv = fgsm_embeddings(model, x, y, criterion, epsilon, clamp_val=clamp_val)
            out_adv, _ = model.rnn(emb_adv)
            logits_adv = model.fc_out(out_adv[:, -1, :])
            loss = criterion(logits_adv, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / max(1, batches)

# -------------------------
# Checkpoint
# -------------------------
def save_checkpoint(model, optimizer, epoch, loss, word_to_ix, ix_to_word, path=CKPT_PATH):
    ckpt = {
        "epoch": int(epoch),
        "loss": float(loss),
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word,
        "seq_len": int(SEQ_LEN),
        "embed_dim": int(EMBED_DIM),
        "hidden_dim": int(HIDDEN_DIM),
        "num_layers": int(NUM_LAYERS),
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path=CKPT_PATH):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"Loaded checkpoint epoch={ckpt['epoch']} loss={ckpt['loss']:.3f}")
    return ckpt

# -------------------------
# Generation (decodes to xaa words)
# -------------------------
@torch.no_grad()
def generate_words(model, word_to_ix, ix_to_word, seed_text, length=50, temp=0.8):
    model.eval()
    V = model.embedding.num_embeddings

    raw = seed_text.strip().split()
    if not raw:
        raw = ["<unk>"]

    # allow either words or ints in seed
    ids = []
    for tok in raw:
        try:
            ids.append(int(tok) % V)
        except ValueError:
            ids.append(word_to_ix.get(tok, 0))

    x0 = torch.tensor([ids], device=device, dtype=torch.long)
    _, h = model(x0, h=None)

    generated_ids = list(ids)
    cur_id = torch.tensor([[ids[-1]]], device=device, dtype=torch.long)

    for _ in range(int(length)):
        logits, h = model.forward_step(cur_id, h=h)
        probs = F.softmax(logits[0] / max(1e-6, float(temp)), dim=-1)
        next_ix = torch.multinomial(probs, 1).item()
        generated_ids.append(next_ix)
        cur_id = torch.tensor([[next_ix]], device=device, dtype=torch.long)

    # decode ids -> words (labels from xaa)
    return " ".join(ix_to_word.get(i, "<unk>") for i in generated_ids)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if not os.path.exists(XAA_PATH):
        raise FileNotFoundError(f"Expected xaa at: {XAA_PATH}")

    ckpt = load_checkpoint(CKPT_PATH)

    if ckpt is not None:
        word_to_ix = ckpt["word_to_ix"]
        ix_to_word = ckpt["ix_to_word"]
        vocab_size = len(word_to_ix)

        model = RNNNextWord(vocab_size, embed_dim=ckpt["embed_dim"], hidden_dim=ckpt["hidden_dim"], num_layers=ckpt["num_layers"]).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        optimizer.load_state_dict(ckpt["optim_state"])
        criterion = nn.CrossEntropyLoss()

        print("Model loaded from checkpoint.")
    else:
        # Build vocab labels from xaa
        word_to_ix, ix_to_word = build_vocab_from_xaa(XAA_PATH, vocab_max=VOCAB_MAX, min_freq=MIN_FREQ)
        vocab_size = len(word_to_ix)
        print(f"xaa vocab_size={vocab_size} (VOCAB_MAX={VOCAB_MAX}, MIN_FREQ={MIN_FREQ})")

        # Build dataset from Arduino analog IDs
        if CACHE_DATASET_PT and os.path.exists(DATASET_PT_PATH):
            d = torch.load(DATASET_PT_PATH, map_location="cpu")
            dataset = TensorDataset(d["X"], d["y"])
            print(f"Loaded cached dataset X={tuple(d['X'].shape)} y={tuple(d['y'].shape)}")
        else:
            ids_t = serial_handshake_and_stream_ids(
                SERIAL_PORT, SERIAL_BAUD,
                vocab_size=vocab_size,
                num_ids=SERIAL_NUM_IDS,
                timeout_s=SERIAL_TIMEOUT_S,
                progress_every=SERIAL_PROGRESS_EVERY
            )
            dataset = make_next_token_dataset_from_ids(ids_t, SEQ_LEN)
            X, y = dataset.tensors
            print(f"Built dataset X={tuple(X.shape)} y={tuple(y.shape)}")

            if CACHE_DATASET_PT:
                torch.save({"X": X, "y": y, "vocab_size": vocab_size, "seq_len": SEQ_LEN}, DATASET_PT_PATH)
                print(f"Saved dataset cache to {DATASET_PT_PATH}")

        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model = RNNNextWord(vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        epsilon = EPS_START
        avg_loss = 0.0

        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training"):
            avg_loss = train_epoch_alternating(
                model, optimizer, criterion, train_loader,
                epsilon=epsilon, adv_every=ADV_EVERY, clamp_val=EMB_CLAMP
            )
            print(f"Epoch {epoch:2d}: avg_loss={avg_loss:.3f}")

            if epoch % EPS_GROW_EVERY == 0:
                epsilon = min(EPS_MAX, epsilon * EPS_GROW_MULT)
                sample = generate_words(model, word_to_ix, ix_to_word, "<unk>", length=30, temp=0.9)
                print(f"  eps={epsilon:.3f} | Sample: {sample}")

        save_checkpoint(model, optimizer, NUM_EPOCHS, avg_loss, word_to_ix, ix_to_word, path=CKPT_PATH)

    print("\nInteractive mode (Ctrl+C to exit).")
    print("Seed can be words (from xaa vocab) or integers (token IDs).")
    while True:
        try:
            cmd = input("SEED: ").strip()
            if not cmd:
                continue
            out = generate_words(model, word_to_ix, ix_to_word, cmd, length=120, temp=100.2)
            print(f"\n{out}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
