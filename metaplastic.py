# metaplastic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import copy

class MetaPlasticCell(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, dt: float = 0.06, smooth_eta: float = 0.86):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dt = dt
        self.smooth_eta = smooth_eta
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.f = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.W_pred = nn.Linear(hidden_dim, input_dim)
        self.dtype = torch.float16
        self.to(self.dtype)
        
        self.logit_feedback_proj = nn.Linear(input_dim, hidden_dim)
        self.feedback_strength = nn.Parameter(torch.tensor(0.08))
        
        # enforce consistent dtype for MPS stability
        self.f = self.f.to(self.dtype)
        self.W_pred = self.W_pred.to(self.dtype)
        self.logit_feedback_proj = self.logit_feedback_proj.to(self.dtype)

        # Multi-timescale persistent memory reservoirs (refactored 3-layer slow field)
        self.fast_memory = nn.Parameter(torch.zeros(1, hidden_dim, device=self._device), requires_grad=False)

        self.slow_base = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=False)
        self.slow_vel  = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=False)
        self.slow_attr = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=False)

        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.memory_gate = self.memory_gate.to(self.dtype)

        self.alpha = nn.Parameter(torch.tensor(0.18))
        self.beta  = nn.Parameter(torch.tensor(0.45))
        self.lambda_decay = nn.Parameter(torch.tensor(0.03))
        
        self.register_buffer('prev_S', torch.zeros(1, hidden_dim, device=self._device, dtype=torch.float16))
        self.register_buffer('prev_dS', torch.zeros(1, hidden_dim, device=self._device, dtype=torch.float16))
        self.register_buffer('meta_trace', torch.zeros(1, hidden_dim, device=self._device, dtype=torch.float16))
        # Persistent state memory (m_t)
        self.register_buffer('m_state', torch.zeros(1, hidden_dim, device=self._device, dtype=torch.float16))
        # per-prompt slow memory anchor (for identity separation)
        self.register_buffer('slow_anchor', torch.zeros(1, hidden_dim, device=self._device, dtype=torch.float16))
        # slow memory adaptability (world drift factor)
        self.register_buffer('slow_epsilon', torch.tensor(0.02))
        self.beta_m = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, h: torch.Tensor, x: torch.Tensor, logits_feedback: torch.Tensor = None, use_meta: bool = True, bank=None):
        device = self.W_pred.weight.device
        dtype = self.W_pred.weight.dtype

        # HARD device sync to prevent CPU/MPS mixing
        fast_memory = self.fast_memory
        slow_base = self.slow_base.to(device=device)
        slow_vel = self.slow_vel.to(device=device)
        slow_attr = self.slow_attr.to(device=device)
        m_state = self.m_state.to(device=device)

        h = h.to(device=device, dtype=dtype)
        x = x.to(device=device, dtype=dtype)

        if logits_feedback is not None:
            logits_feedback = logits_feedback.to(device=device, dtype=dtype)

        S = F.silu(h)
        raw_dS = (S - self.prev_S) / self.dt
        dS_smooth = self.smooth_eta * self.prev_dS + (1 - self.smooth_eta) * raw_dS
        
        x_hat = self.W_pred(h)
        error = x - x_hat

        fx = self.f(torch.cat([h, x], dim=1))
        decay = torch.exp(-self.lambda_decay.abs() * self.dt)

        # Persistent memory reservoirs
        mem_inp = torch.cat(
            [h, (slow_base + 1.1 * slow_vel + 0.35 * slow_attr).expand_as(h)],
            dim=1
        ).to(dtype)

        mem_gate = self.memory_gate(mem_inp)

        fast_memory = 0.82 * fast_memory + 0.18 * h.mean(dim=0, keepdim=True)

        noise = torch.randn_like(fast_memory) * 0.005

        decorrelated_fast = fast_memory - 0.5 * noise
        decorrelated_dS = dS_smooth.mean(dim=0, keepdim=True) + noise

        with torch.no_grad():
            self.fast_memory.copy_(fast_memory)

        h_mean = h.mean(dim=0, keepdim=True)
        prompt_signature = torch.tanh(h_mean - (bank['slow_base'] if bank is not None else slow_base))

        steps = bank['steps'] if bank is not None else 0
        with torch.no_grad():
            # holographic slow memory update (non-collapsing phase navigation)
            phase = h_mean - m_state
            phase = phase / (phase.norm(dim=-1, keepdim=True) + 1e-8)

            holographic_update = (
                slow_base
                + 0.03 * prompt_signature
                + 0.01 * torch.randn_like(slow_base)
                + 0.01 * phase
            )

            # freeze holographic update during early steps
            if steps <= 15:
                pass
            else:
                slow_base.copy_(
                    F.normalize(holographic_update, dim=-1)
                    * slow_base.norm(dim=-1, keepdim=True)
                )
        delta = h_mean - (bank['slow_base'] if bank is not None else slow_base)

        with torch.no_grad():
            slow_vel.copy_(
                0.98 * slow_vel + 0.02 * torch.sign(delta) * torch.abs(delta)
            )
        with torch.no_grad():
            slow_vel.add_(0.02 * torch.randn_like(slow_vel))

        with torch.no_grad():
            slow_attr.copy_(
                0.985 * slow_attr + 0.015 * torch.tanh(h_mean)
            )
        base_ref = bank['slow_base'] if bank is not None else slow_base
        steps = bank['steps'] if bank is not None else 0
        gate = torch.tensor(min(steps / 20.0, 1.0), device=h.device, dtype=h.dtype)

        # ── soft-freeze slow drift ──
        if steps <= 15:
            slow_base_effective = slow_base.detach()
        else:
            slow_base_effective = bank['slow_base']

        base_ref = slow_base_effective if bank is not None else slow_base

        memory_context = mem_gate * gate * base_ref.expand_as(h)

        # persistent state memory (m_t update)
        with torch.no_grad():
            m_state.copy_(
                (1 - torch.sigmoid(self.beta_m)) * m_state +
                torch.sigmoid(self.beta_m) * h.mean(dim=0, keepdim=True)
            )

        # === ECHO LOOP: logits feedback into dynamics (gated) ===
        if logits_feedback is not None:
            fb = self.logit_feedback_proj(logits_feedback)

            fb_gate = torch.sigmoid(
                self.feedback_strength * dS_smooth.norm(dim=1, keepdim=True) - 1.8
            )

            memory_context = memory_context + fb_gate * fb
        
        # meta dynamics with temporal accumulation (directional memory)
        if use_meta:
            memory_gain = torch.sigmoid((h_mean - (bank['slow_base'] if bank is not None else slow_base)).norm(dim=-1, keepdim=True))
            slow_comb = ((bank['slow_base'] if bank is not None else slow_base) + 1.1 * (bank['slow_vel'] if bank is not None else slow_vel) + 0.35 * (bank['slow_attr'] if bank is not None else slow_attr)) * memory_gain

            contrast = dS_smooth - slow_comb.expand_as(dS_smooth)

            self.meta_trace = 0.7 * self.meta_trace + 0.3 * contrast

            # --- phase-space orthogonalized meta dynamics ---
            meta_dir = self.meta_trace

            proj = (meta_dir * dS_smooth).sum(dim=-1, keepdim=True) * dS_smooth
            meta_dir = meta_dir - 0.6 * proj + 0.2 * torch.randn_like(meta_dir)

            meta_gain = 1.8 * torch.sigmoid(self.beta_m)
            meta_term = meta_gain * self.beta * meta_dir
        else:
            meta_term = torch.zeros_like(h)

        h_new = decay * h + (1 - decay) * fx + \
                self.alpha * torch.matmul(error, self.W_pred.weight.to(dtype)) + \
                0.35 * memory_context + \
                0.25 * m_state.expand_as(h) + \
                meta_term

        # Softer bounded dynamics with residual openness
        h_new = 0.72 * h + 0.28 * torch.tanh(h_new)

        # soft amplitude gating instead of hard clamp
        h_norm = h_new.norm(dim=-1, keepdim=True) + 1e-6
        h_new = h_new * torch.sigmoid(2.0 - h_norm)
        
        self.prev_S.copy_(S.detach())
        self.prev_dS.copy_(dS_smooth.detach())
        
        return h_new


class MetaPlasticProber(nn.Module):
    def __init__(self, hidden_dim=384, num_layers=3, model_name: str = "gpt2"):
        super().__init__()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        if "mistral" in self.model_name.lower():
            self.base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                output_hidden_states=True
            ).to(self.device)
        else:
            self.base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                output_hidden_states=True
            ).to(self.device)

        self.base.eval()
        if "gpt2" in self.model_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # align custom layers to HF device (device_map="auto" compatible)
        self._device = self.device

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        gpt_hidden = self.base.config.hidden_size

        self.input_proj = nn.Linear(gpt_hidden, hidden_dim).to(self._device).to(torch.float16)
        self.meta_to_hidden = nn.Linear(hidden_dim, gpt_hidden).to(self._device).to(torch.float16)

        self.cells = nn.ModuleList([
            MetaPlasticCell(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # IMPORTANT: ensure meta cells are on same device as base model (fixes CPU/MPS mismatch)
        for cell in self.cells:
            cell.to(self._device)

        self.norm = nn.LayerNorm(hidden_dim).to(self._device).to(torch.float16)
        self.hidden_state = None
        self._warmed = False
        self._warming = False
        # dynamical temperature (controls phase transitions in hidden state)
        self.T_dyn = nn.Parameter(torch.tensor(1.0))
        self.time_lambda = nn.Parameter(torch.tensor(0.15))
        self.register_buffer('running_variance', torch.tensor(0.003, device=self._device))
        # per-prompt slow memory bank (prevents global attractor collapse)
        self.slow_base_bank = {}
    def warmup(self, input_ids):
        with torch.no_grad():
            # run base model only (avoid recursive self.forward call)
            outputs = self.base(
                input_ids.to(self.device),
                output_hidden_states=True,
                use_cache=False
            )

            base_hidden = outputs.hidden_states[-1].to(self._device).to(torch.float16)
            x = self.input_proj(base_hidden)

            # minimal single-step initialization of meta cells
            h_list = [torch.zeros(input_ids.shape[0], self.hidden_dim,
                                  device=self._device, dtype=torch.float16)
                      for _ in range(self.num_layers)]

            current_x = x[:, -1, :]

            for i, cell in enumerate(self.cells):
                inp = current_x if i == 0 else h_list[i - 1]
                h_list[i] = cell(h_list[i], inp, logits_feedback=None, use_meta=True)

            slow_mem = (
                self.cells[-1].slow_base +
                0.7 * self.cells[-1].slow_vel +
                0.5 * self.cells[-1].slow_attr
            ).detach()

            self.slow_origin = slow_mem.clone()
            self.prev_slow_mem = slow_mem.clone()
            # trajectory imprint update (instead of hard overwrite)
            self.cells[-1].slow_anchor.copy_(
                0.7 * self.cells[-1].slow_anchor + 0.3 * slow_mem
            )

        self._warmed = True
        self._warming = False

    def reset_hidden(self, batch_size=1, device=None):
        # ALWAYS force model device to avoid CPU/MPS mixing bugs
        device = self.device

        self.hidden_state = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=torch.float16)
            for _ in range(self.num_layers)
        ]

        for cell in self.cells:
            cell.fast_memory.data.zero_()
            cell.slow_base.data.zero_()
            cell.slow_vel.data.zero_()
            cell.slow_attr.data.zero_()
            cell.prev_S.zero_()
            cell.prev_dS.zero_()
            cell.meta_trace.zero_()
            cell.m_state.zero_()
            # keep slow_anchor persistent across resets (trajectory memory)
        # IMPORTANT: ensure slow_anchor does not leak across different generation runs

        if hasattr(self, "past_key_values"):
            self.past_key_values = None

    def forward(self, input_ids, use_meta=True):
        input_ids = input_ids.to(self.device)
        # ================= PER-PROMPT SLOW MEMORY ISOLATION =================
        prompt_key = hash(tuple(input_ids[0].tolist()))

        if prompt_key not in self.slow_base_bank:
            # initialize per-prompt slow state + full slow manifold (decoupled dynamics)
            self.slow_base_bank[prompt_key] = {
                'slow_base': self.cells[-1].slow_base.clone().detach(),
                'slow_vel': self.cells[-1].slow_vel.clone().detach(),
                'slow_attr': self.cells[-1].slow_attr.clone().detach(),
                'slow_anchor': self.cells[-1].slow_anchor.clone().detach(),
                'drift_vec': (torch.randn_like(self.cells[-1].slow_base) * 0.01),
                'steps': 0
            }

        bank = self.slow_base_bank[prompt_key]
        bank['steps'] += 1
        # ===============================================================

        if self.hidden_state is None:
            self.reset_hidden(input_ids.shape[0])

        outputs = self.base(
            input_ids.to(self.device),
            output_hidden_states=True,
            use_cache=False
        )

        target_device = self.device
        base_hidden = outputs.hidden_states[-1].to(self._device).to(torch.float16)
        base_logits = outputs.logits

        x = self.input_proj(base_hidden)
        h_list = self.hidden_state

        logits_feedback_acc = None

        # ensure warmup phase is done (avoid recursion)
        if (not self._warmed) and (not getattr(self, "_warming", False)):
            self._warming = True
            self.warmup(input_ids)

        # SINGLE-STEP META UPDATE (time-aligned with generation step)
        current_x = x[:, -1, :]

        # simple echo feedback: use current projected hidden state minus slow_base as stable signal
        fb = current_x - self.cells[0].slow_base

        # (removed: per-step injection of slow state into all cells)

        for i, cell in enumerate(self.cells):
            inp = current_x if i == 0 else h_list[i - 1]
            h_new = cell(h_list[i], inp, logits_feedback=fb, use_meta=use_meta, bank=bank)
            mix = 0.75 + 0.1 * torch.tanh(self.T_dyn - 1.0)
            h_list[i] = mix * h_list[i] + (1 - mix) * h_new

        logits_feedback_acc = current_x  # use projected hidden as proxy feedback
        logits_feedback_acc = current_x

        # (removed: restore of original shared parameters)

        # phase-transition temperature scaling (stable, aligned timescale)
        if hasattr(self, "T_dyn"):
            with torch.no_grad():
                h_stack = torch.stack(h_list, dim=0).float()
                current_variance = h_stack.std(dim=0).mean()

                # use running variance if available, fallback to current
                if hasattr(self, "running_variance"):
                    current_var = torch.tensor(self.running_variance.item(), device=self.T_dyn.device)
                else:
                    current_var = current_variance

                target_variance = 0.005
                error_var = current_var - target_variance

                self.T_dyn.data -= 0.005 * error_var
                self.T_dyn.data.clamp_(0.5, 2.0)

            # controlled exploration noise (causal-memory interference time field)
            bank = self.slow_base_bank[prompt_key]
            slow_comb = (
                0.85 * bank['slow_base'] +
                0.7 * bank['slow_vel'] +
                0.5 * bank['slow_attr'] +
                0.1 * bank['drift_vec']
            )

            # Optionally, adjust for meta if enabled
            if use_meta:
                slow_comb = slow_comb - 0.05 * torch.tanh(bank['drift_vec'])

            causal_time = current_x
            memory_time = slow_comb.expand_as(current_x)

            phase_coupling = F.cosine_similarity(
                causal_time,
                memory_time,
                dim=-1
            ).mean()

            T_eff = self.T_dyn + self.time_lambda * phase_coupling
            noise_scale = (T_eff - 1.0).clamp(0.0) * 0.01

            for i in range(len(h_list)):
                h_list[i] = h_list[i] + noise_scale * torch.randn_like(h_list[i])

        # write-back isolated slow_base to bank (prevents cross-prompt contamination)
        slow_comb = (
            self.cells[-1].slow_base +
            0.7 * self.cells[-1].slow_vel +
            0.5 * self.cells[-1].slow_attr
        )
        with torch.no_grad():
            drift = bank['drift_vec']

            # drift ALWAYS updates
            drift.mul_(0.997).add_(0.003 * torch.randn_like(drift))

            # delayed slow-manifold write-back
            if bank['steps'] > 15:
                bank['slow_base'].mul_(0.995).add_(0.005 * self.cells[-1].slow_base + drift)
                bank['slow_vel'].mul_(0.99).add_(0.01 * self.cells[-1].slow_vel)
                bank['slow_attr'].mul_(0.99).add_(0.01 * self.cells[-1].slow_attr)

            # anchor always tracks global trajectory
            bank['slow_anchor'].copy_(
                0.85 * bank['slow_anchor'] + 0.15 * slow_comb.detach()
            )

        self.hidden_state = h_list

        meta_hidden = self.norm(h_list[-1].to(self._device).to(torch.float16))
        meta_proj = self.meta_to_hidden(meta_hidden).unsqueeze(1)

        if use_meta:
            meta_logits = self.base.lm_head(meta_proj)
            logits = base_logits[:, -1:, :] * 0.92 + meta_logits * 0.08
        else:
            logits = base_logits[:, -1:, :]

        return logits


# ====================== PROBING ======================

# NOTE:
# logits_feedback creates a closed causal loop:
# tokens → hidden → meta-dynamics → logits → feedback into meta-dynamics
# This makes the system non-Markovian in hidden space.
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    model = MetaPlasticProber(hidden_dim=384, num_layers=3)

    model.eval()
    
    tokenizer = model.tokenizer
    
    prompt = "The most beautiful thing about consciousness is"
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    print("=== PROBING DYNAMICS ===\n")
    
    def generate_with_mode(use_meta: bool, name: str, max_new=60):
        # local tracking state for slow memory drift (must be defined per run)
        slow_mem_origin = None
        prev_slow_mem = None
        model.reset_hidden()
        generated = input_ids.clone()
        h_norms = []
        h_history = []
        entropies = []
        prev_h = None
        
        print(f"\n--- {name} (meta={use_meta}) ---")
        
        for i in range(max_new):
            with torch.no_grad():
                logits = model(generated, use_meta=use_meta)

                temperature = 0.9
                top_k = 50

                scaled_logits = logits[:, -1, :] / temperature

                # top-k filtering
                values, indices = torch.topk(scaled_logits, top_k)
                filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                filtered_logits.scatter_(1, indices, values)

                probs = F.softmax(filtered_logits, dim=-1)

                # --- ANTI-COLLAPSE ENTROPY FLOOR ---
                entropy_tensor = - (probs * torch.log(probs + 1e-9)).sum(dim=-1, keepdim=True)
                min_entropy = 0.6

                collapse_mask = (entropy_tensor < min_entropy).float()

                noise = torch.rand_like(probs)
                probs = (1 - collapse_mask) * probs + collapse_mask * (0.85 * probs + 0.15 * noise)

                probs = probs / probs.sum(dim=-1, keepdim=True)

                entropy = - (probs * torch.log(probs + 1e-9)).sum(-1).mean().item()
                entropies.append(entropy)
                # detect collapse states
                if entropy < 0.1:
                    top_token_id = logits[:, -1, :].argmax(-1).item()
                    top_token = tokenizer.decode([top_token_id])
                    print(f"  *** COLLAPSE at step {i}: token='{top_token}', entropy={entropy:.4f}")

                # stochastic sampling instead of frozen argmax attractor
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                current_h = torch.stack(model.hidden_state, dim=0).mean(dim=0).detach()
                h_history.append(current_h.detach().cpu())
                if len(h_history) > 5:
                    h_stack = torch.stack(h_history[-10:], dim=0)
                    current_var = h_stack.std(dim=0).mean().item()

                    with torch.no_grad():
                        model.running_variance.copy_(torch.tensor(current_var, device=model.device))
                meta_strength = dS_norm = None
                if use_meta:
                    dS_norm = model.cells[-1].prev_dS.norm().item()
                    meta_strength = (model.cells[-1].beta * model.cells[-1].prev_dS).norm().item()
                    slow_mem_norm = (
                        model.cells[-1].slow_base +
                        0.7 * model.cells[-1].slow_vel +
                        0.5 * model.cells[-1].slow_attr
                    ).norm().item()
                h_norms.append(current_h.norm().item())

                slow_mem = (
                    model.cells[-1].slow_base +
                    0.7 * model.cells[-1].slow_vel +
                    0.5 * model.cells[-1].slow_attr
                ).detach()

                # initialize origin AFTER first valid slow state
                if slow_mem_origin is None or prev_slow_mem is None:
                    slow_mem_origin = slow_mem.clone().detach()
                    prev_slow_mem = slow_mem.clone().detach()
                    drift_deg = 0.0
                    step_deg = 0.0
                else:
                    # cumulative drift from origin (degrees)
                    cos_from_origin = F.cosine_similarity(
                        slow_mem.float().to(model.device),
                        slow_mem_origin.float().to(model.device)
                    )
                    cos_from_origin = cos_from_origin.clamp(-1.0, 1.0).item()
                    drift_deg = np.degrees(np.arccos(cos_from_origin))

                    # stepwise angular velocity (degrees per step)
                    cos_step = F.cosine_similarity(
                        slow_mem.float().to(model.device),
                        prev_slow_mem.float().to(model.device)
                    )
                    cos_step = cos_step.clamp(-1.0, 1.0).item()
                    step_deg = np.degrees(np.arccos(cos_step))

                    prev_slow_mem = slow_mem.clone().detach()
                
                if prev_h is not None:
                    h1 = F.normalize(current_h, dim=-1)
                    h2 = F.normalize(prev_h, dim=-1)
                    cos = (h1 * h2).sum(dim=-1).mean().item()
                    if i % 12 == 0:
                        if use_meta:
                            print(
                                f"Step {i:2d} | Entropy: {entropy:.3f} | "
                                f"Cos drift: {cos:.4f} | h_norm: {h_norms[-1]:.2f} | "
                                f"dS_norm: {dS_norm:.4f} | meta_norm: {meta_strength:.4f} | "
                                f"slow_drift: {drift_deg:.3f} | slow_vel: {step_deg:.4f} | T_dyn: {model.T_dyn.item():.3f}"
                            )
                        else:
                            print(
                                f"Step {i:2d} | Entropy: {entropy:.3f} | "
                                f"Cos drift: {cos:.4f} | h_norm: {h_norms[-1]:.2f}"
                            )
                prev_h = current_h.detach().clone()
        
        text = tokenizer.decode(generated[0].cpu().tolist())
        print(f"\nGenerated: {text[-180:]}")
        order_param = None
        if len(h_history) > 5:
            h_stack = torch.stack(h_history, dim=0)
            order_param = h_stack.std(dim=0).mean().item()
            print(f"\nORDER PARAMETER (variance): {order_param:.6f}")
        return entropies, h_norms
    
    def compare_histories(prompt_a: str, prompt_b: str, use_meta=True, steps=40):
        print("\n" + "=" * 70)
        print(f"HISTORY DIVERGENCE TEST | meta={use_meta}")

        ids_a = torch.tensor([tokenizer.encode(prompt_a)], device=device)
        ids_b = torch.tensor([tokenizer.encode(prompt_b)], device=device)

        model.reset_hidden()
        generated_a = ids_a.clone()

        traj_a = []

        for _ in range(steps):
            with torch.no_grad():
                logits = model(generated_a, use_meta=use_meta)
                probs = F.softmax(logits[:, -1, :] / 0.9, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                generated_a = torch.cat([generated_a, token], dim=1)
                traj_a.append(model.hidden_state[-1].detach().clone())

        hidden_snapshot = [h.detach().to(model.device).clone() for h in model.hidden_state]
        prev_S_snapshot = [cell.prev_S.detach().to(model.device).clone() for cell in model.cells]
        prev_dS_snapshot = [cell.prev_dS.detach().to(model.device).clone() for cell in model.cells]

        model.reset_hidden()
        generated_b = ids_b.clone()

        traj_b = []

        for _ in range(steps):
            with torch.no_grad():
                logits = model(generated_b, use_meta=use_meta)
                probs = F.softmax(logits[:, -1, :] / 0.9, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                generated_b = torch.cat([generated_b, token], dim=1)
                traj_b.append(model.hidden_state[-1].detach().clone())

        divergences = []

        for ha, hb in zip(traj_a, traj_b):
            cos = F.cosine_similarity(ha, hb).item()
            divergences.append(1.0 - cos)

        print(f"Prompt A: {prompt_a}")
        print(f"Prompt B: {prompt_b}")
        print(f"Initial divergence : {divergences[0]:.6f}")
        print(f"Final divergence   : {divergences[-1]:.6f}")
        print(f"Mean divergence    : {np.mean(divergences):.6f}")

        print("\nTrajectory divergence curve:")
        for i in range(0, len(divergences), 5):
            print(f"Step {i:2d} -> {divergences[i]:.6f}")
    
    # Сравнение
    ent_meta, norm_meta = generate_with_mode(True,  "WITH Meta-Plasticity")
    ent_base, norm_base = generate_with_mode(False, "WITHOUT Meta-Plasticity")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print(f"Avg entropy with meta    : {np.mean(ent_meta):.3f}")
    print(f"Avg entropy without meta : {np.mean(ent_base):.3f}")
    print(f"Final h_norm with meta   : {norm_meta[-1]:.2f}")
    print(f"Final h_norm without     : {norm_base[-1]:.2f}")

    compare_histories(
        "The most beautiful thing about consciousness is",
        "Violence emerges when memory collapses into fear",
        use_meta=True
    )

    compare_histories(
        "The most beautiful thing about consciousness is",
        "Violence emerges when memory collapses into fear",
        use_meta=False
    )
