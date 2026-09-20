"""Actual forward/backward/AdamW on a tiny causal Transformer. CPU/offline reference.

Default data are algorithmically generated token sequences, NOT natural-language
pretraining data. A local UTF-8 corpus is optional. No model or data are downloaded.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .io import JSONLLogger, write_json
from .monitor import MonitorConfig, OnlineMonitor
from .evaluation import observed_loss_events, evaluate
from .benchmark import environment


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, width: int = 48, heads: int = 4, layers: int = 2, context: int = 24):
        super().__init__()
        if vocab < 2 or width % heads or layers < 1 or context < 2:
            raise ValueError("Invalid Transformer dimensions")
        self.context = context
        self.embedding = nn.Embedding(vocab,width)
        self.position = nn.Embedding(context,width)
        # Separately initialized layers; encoder-style self-attention is explicitly causal.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width,heads,2*width,dropout=0.0,
                                       activation="gelu",batch_first=True,norm_first=True)
            for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width,vocab,bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.context:
            raise ValueError("Expected batch x sequence with sequence <= context")
        length = tokens.shape[1]
        x = self.embedding(tokens)+self.position(torch.arange(length,device=tokens.device))[None,:,:]
        mask = torch.triu(torch.ones(length,length,dtype=torch.bool,device=tokens.device),diagonal=1)
        for block in self.blocks:
            x = block(x,src_mask=mask)
        return self.head(self.norm(x))


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 320
    batch_size: int = 12
    context: int = 24
    vocab: int = 32
    width: int = 48
    layers: int = 2
    heads: int = 4
    learning_rate: float = .002
    seed: int = 7
    scenario: str = "normal"
    fault_start: int = 190
    fault_ramp: int = 30
    fault_hold: int = 25
    fault_multiplier: float = 50.0
    clip_norm: float = 1.0
    threads: int = 1

    def __post_init__(self):
        if self.steps < 10 or self.batch_size < 1 or self.context < 2 or self.learning_rate <= 0:
            raise ValueError("Invalid training scale or learning rate")
        if self.scenario not in ("normal","lr_fault"):
            raise ValueError("scenario must be normal or lr_fault")
        if self.clip_norm <= 0 or self.threads < 1 or self.fault_ramp < 1 or self.fault_hold < 0 or self.fault_multiplier <= 0:
            raise ValueError("Invalid clipping, threads or fault configuration")
        if self.scenario == "lr_fault" and not 1 <= self.fault_start < self.steps:
            raise ValueError("fault_start must occur within training")


def parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    """Full parameter copy: O(number of parameters) extra memory. Tiny-demo use only."""
    return [p.detach().clone() for p in model.parameters()]


def measured_parameter_change(model: nn.Module, before: list[torch.Tensor]) -> tuple[float,float]:
    current = list(model.parameters())
    if len(current) != len(before) or any(a.shape != b.shape for a,b in zip(current,before)):
        raise ValueError("Snapshot and model do not match")
    with torch.no_grad():
        update_squared = sum(((p.detach()-b).float()**2).sum() for p,b in zip(current,before))
        param_squared = sum((b.float()**2).sum() for b in before)
    return float(torch.sqrt(update_squared)),float(torch.sqrt(param_squared))


def gradient_norm(model: nn.Module) -> float:
    gradients = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    if not gradients:
        return 0.0
    return float(torch.sqrt(sum((g*g).sum() for g in gradients)))


class TokenSource:
    def __init__(self, cfg: TrainConfig, corpus: str | Path | None = None):
        self.cfg = cfg
        self.train_rng = torch.Generator(device="cpu").manual_seed(cfg.seed+1000)
        self.validation_rng = torch.Generator(device="cpu").manual_seed(cfg.seed+100000)
        self.corpus = None
        self.vocab = cfg.vocab
        self.metadata = {"data_kind": "algorithmic_token_sequences", "vocab": self.vocab,
                         "description": "Uniform start tokens and arithmetic next-token transitions modulo vocabulary; one fixed increment (1) per sequence."}
        if corpus is not None:
            path = Path(corpus)
            text = path.read_text(encoding="utf-8")
            chars = sorted(set(text))
            if len(chars) < 2 or len(chars) > 4096:
                raise ValueError("Local corpus needs 2..4096 unique characters")
            if len(text) < 20*(cfg.context+1):
                raise ValueError("Local corpus too short for an 80/20 train/validation split")
            mapping = {ch:i for i,ch in enumerate(chars)}
            tokens = torch.tensor([mapping[ch] for ch in text],dtype=torch.long)
            split = int(.8*len(tokens))
            self.corpus = (tokens[:split],tokens[split:])
            self.vocab = len(chars)
            self.metadata = {"data_kind": "user_local_utf8_corpus", "characters": len(text),
                             "vocab": self.vocab, "train_fraction": .8,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "note": "Vocabulary is built from the supplied corpus; no external data or checkpoints."}

    def batch(self, validation: bool = False):
        cfg = self.cfg
        rng = self.validation_rng if validation else self.train_rng
        if self.corpus is None:
            starts = torch.randint(self.vocab,(cfg.batch_size,1),generator=rng)
            tokens = (starts+torch.arange(cfg.context+1)[None,:]) % self.vocab
        else:
            pool = self.corpus[int(validation)]
            starts = torch.randint(len(pool)-cfg.context,(cfg.batch_size,),generator=rng)
            tokens = torch.stack([pool[int(i):int(i)+cfg.context+1] for i in starts])
        return tokens[:,:-1],tokens[:,1:]


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if cfg.scenario == "normal" or step < cfg.fault_start:
        return cfg.learning_rate
    elapsed = step-cfg.fault_start
    if elapsed < cfg.fault_ramp:
        return cfg.learning_rate*cfg.fault_multiplier**((elapsed+1)/cfg.fault_ramp)
    if elapsed < cfg.fault_ramp+cfg.fault_hold:
        return cfg.learning_rate*cfg.fault_multiplier
    return cfg.learning_rate


def train(out: str | Path, cfg: TrainConfig | None = None,
          monitor_config: MonitorConfig | None = None, corpus: str | Path | None = None) -> dict:
    cfg = cfg or TrainConfig()
    conf = monitor_config or MonitorConfig()
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"telemetry.jsonl").exists():
        raise FileExistsError("Output run already exists; use a new directory (no automatic overwrite)")
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    source = TokenSource(cfg,corpus)
    model = TinyCausalLM(source.vocab,cfg.width,cfg.heads,cfg.layers,cfg.context)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=.01)
    monitor = OnlineMonitor(conf)
    run_id = f"tiny-transformer-{cfg.scenario}-{cfg.seed}"
    params = sum(p.numel() for p in model.parameters())
    write_json(out/"config.json", {"training": asdict(cfg), "monitor": conf.to_dict(),
                                    "source": source.metadata, "run_id": run_id, "parameters": params,
                                    "environment": environment(), "device": "cpu", "precision": "float32"})
    validation_x,validation_y = source.batch(validation=True)
    def val_loss():
        model.eval()
        with torch.no_grad():
            value = float(F.cross_entropy(model(validation_x).reshape(-1,source.vocab),validation_y.reshape(-1)))
        model.train()
        return value
    initial_val = val_loss()
    telemetry, scores = [], []
    probe_seconds, monitor_seconds, write_seconds = 0.0,0.0,0.0
    started = time.perf_counter()
    with JSONLLogger(out/"telemetry.jsonl") as logger:
        for step in range(cfg.steps):
            lr = _lr_at(step,cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x,y = source.batch()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).reshape(-1,source.vocab),y.reshape(-1))
            loss.backward()
            # Returned total_norm is measured before clipping. AMP needs unscale_ first.
            raw_grad = float(nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_norm,error_if_nonfinite=False))
            finite = math.isfinite(float(loss.detach())) and math.isfinite(raw_grad)
            begin = time.perf_counter()
            clipped = gradient_norm(model)
            before = parameter_snapshot(model)
            probe_seconds += time.perf_counter()-begin
            if finite:
                optimizer.step()
            begin = time.perf_counter()
            update,param = measured_parameter_change(model,before)
            probe_seconds += time.perf_counter()-begin
            row = {"step": step,"run_id":run_id,"loss":float(loss.detach()),
                   "grad_norm":raw_grad,"clipped_grad_norm":clipped,"learning_rate":lr,
                   "update_norm":update,"param_norm":param,"tokens":cfg.batch_size*cfg.context,
                   "optimizer_step_applied":finite}
            begin = time.perf_counter()
            diagnostic = monitor.update(row)
            monitor_seconds += time.perf_counter()-begin
            begin = time.perf_counter()
            logger.write(row)
            write_seconds += time.perf_counter()-begin
            telemetry.append(row)
            scores.append(diagnostic)
            if not finite:
                break  # Log failure, do NOT silently propagate corrupted parameters.
    training_seconds = time.perf_counter()-started
    final_val = val_loss()
    frame,scored = pd.DataFrame(telemetry),pd.DataFrame(scores)
    frame.to_csv(out/"telemetry.csv",index=False)
    scored.to_csv(out/"scores.csv",index=False)
    events = observed_loss_events(frame)
    write_json(out/"observed_events.json",{"events":events,"rule":{"window":64,"min_rise":.7,"ratio":1.8,"persistence":2},
                                           "note":"Labels derived from measured loss, NOT the intervention schedule or warning time."})
    metrics = evaluate(scored,events,horizon=40)
    summary = {"run_id":run_id,"scenario":cfg.scenario,"seed":cfg.seed,"parameters":params,
               "records":len(frame),"optimizer_updates_applied":int(frame["optimizer_step_applied"].sum()),
               "stopped_for_nonfinite":bool(not frame["optimizer_step_applied"].iloc[-1]),"tokens_processed":len(frame)*cfg.batch_size*cfg.context,
               "initial_validation_loss":initial_val,"final_validation_loss":final_val,
               "training_loop_seconds":training_seconds,"parameter_probe_seconds":probe_seconds,
               "monitor_seconds":monitor_seconds,"jsonl_write_seconds":write_seconds,
               "monitor_fraction_of_instrumented_loop":monitor_seconds/training_seconds,
               "all_gradients_finite":bool(np.isfinite(frame["grad_norm"]).all()),
               "data":source.metadata, "metrics":metrics,
               "scope":"CPU tiny-model integration experiment; not a large-model or natural-language quality benchmark.",
               "overhead_caveat":"Component timing inside an instrumented run, NOT paired baseline slowdown. Full parameter snapshots cost O(parameters)."}
    write_json(out/"summary.json",summary)
    # Save own state dict only, without claiming a production checkpoint/recovery system.
    torch.save(model.state_dict(),out/"final_state_dict.pt")
    return summary
