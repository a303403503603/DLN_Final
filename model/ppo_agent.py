"""PPO Agent - 多股票交易的增強學習代理"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
from pipeline.config import INPUT_DIM, OUTPUT_DIM, PPO, CANDIDATE_POOL_SIZE, PER_STOCK_DIM

EPS = 1e-8


class PPOActorCritic(nn.Module):
    """Per-stock 架構：輸出 allocation weights（含現金桶）"""
    def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                 n_stocks: int = CANDIDATE_POOL_SIZE,
                 per_stock_dim: int = PER_STOCK_DIM):
        super().__init__()
        self.n_stocks = n_stocks
        self.per_stock_dim = per_stock_dim

        # Per-stock encoder（權重共用）
        self.stock_encoder = nn.Sequential(
            nn.Linear(per_stock_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor heads: per-stock score + cash score
        self.actor_head = nn.Sequential(
            nn.Linear(128, 1),
        )
        self.cash_head = nn.Sequential(
            nn.Linear(128, 1),
        )

        # Critic：aggregate 所有 stock 特徵 → 預測 portfolio value
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _forward_features(self, x):
        """x: (batch, OBS_TOTAL) → per-stock embeddings + pooled + actor logits"""
        b = x.shape[0]
        x = x.reshape(b, self.n_stocks, self.per_stock_dim)
        feats = self.stock_encoder(x.reshape(-1, self.per_stock_dim))
        feats = feats.reshape(b, self.n_stocks, -1)
        stock_logits = self.actor_head(feats.reshape(-1, 128))
        stock_logits = stock_logits.reshape(b, self.n_stocks)

        pooled = feats.mean(dim=1)
        cash_logit = self.cash_head(pooled)

        logits = torch.cat([stock_logits, cash_logit], dim=1)
        return feats, pooled, logits

    def forward(self, x: torch.Tensor, stochastic: bool = True):
        stock_feats, pooled, actor_logits = self._forward_features(x)

        alpha = F.softplus(actor_logits) + EPS
        dist = Dirichlet(alpha)
        if stochastic:
            actions = dist.sample()
        else:
            actions = alpha / alpha.sum(dim=1, keepdim=True)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        state_value = self.critic(pooled).squeeze(-1)

        if not stochastic:
            entropy = 0.0

        return actions, log_probs, state_value, entropy

    def evaluate(self, x, given_actions):
        stock_feats, pooled, actor_logits = self._forward_features(x)
        alpha = F.softplus(actor_logits) + EPS
        dist = Dirichlet(alpha)

        actions = torch.clamp(given_actions, EPS, 1.0)
        actions = actions / actions.sum(dim=1, keepdim=True)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        state_value = self.critic(pooled).squeeze(-1)

        return log_probs, state_value, entropy


class PPOAgent:
    """PPO Agent wrapper"""
    def __init__(self, input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM,
                 learning_rate: float = PPO['learning_rate']):
        self.actor_critic = PPOActorCritic(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor_critic = self.actor_critic.to(self.device)
        if self.device == 'cuda':
            try:
                self.actor_critic = torch.compile(self.actor_critic, mode='reduce-overhead')
            except Exception:
                pass

    def update(self, states, actions, log_probs, returns, advantages):
        """PPO policy update with per-stock ratio."""
        self.actor_critic.train()

        new_log_probs, state_values, entropy = self.actor_critic.evaluate(states, actions)

        if torch.isnan(new_log_probs).any():
            print("[NaN DEBUG] new_log_probs has NaN")
            print(f"  states has NaN: {torch.isnan(states).any()}")
            print(f"  actions has NaN: {torch.isnan(actions).any()}")
            return {'policy_loss': float('nan'), 'value_loss': float('nan'), 'entropy': 0.0}

        if torch.isnan(log_probs).any():
            print("[NaN DEBUG] stored log_probs has NaN")
            return {'policy_loss': float('nan'), 'value_loss': float('nan'), 'entropy': 0.0}

        # Ratio: exp(new_log_prob - old_log_prob) per batch
        ratios = torch.exp(new_log_probs - log_probs)  # (batch,)

        if torch.isnan(ratios).any() or torch.isinf(ratios).any():
            print(f"[NaN DEBUG] ratios has NaN/inf")
            print(f"  new_log_probs range: [{new_log_probs.min():.4f}, {new_log_probs.max():.4f}]")
            print(f"  log_probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
            ratios = torch.nan_to_num(ratios, nan=1.0, posinf=2.0, neginf=0.0)

        # PPO Clip objective — advantages: (batch,) scalar or (batch, n_stocks) per-stock
        adv = advantages.view(-1)
        clipped_ratios = torch.clamp(ratios, 1 - PPO['clip_range'], 1 + PPO['clip_range'])
        policy_loss_1 = ratios * adv
        policy_loss_2 = clipped_ratios * adv
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Value loss (Huber)
        value_loss = nn.SmoothL1Loss()(state_values, returns)

        # Total loss
        total_loss = policy_loss + PPO['vf_coef'] * value_loss + PPO['ent_coef'] * entropy

        if torch.isnan(total_loss).any():
            print(f"[NaN DEBUG] total_loss is NaN")
            print(f"  policy_loss: {policy_loss}, value_loss: {value_loss}, entropy: {entropy}")
            return {'policy_loss': float('nan'), 'value_loss': float('nan'), 'entropy': 0.0}

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO['max_grad_norm'])
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }

    def get_action(self, state: np.ndarray, stochastic: bool = False):
        """Single inference."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            actions, log_probs, values, _ = self.actor_critic(state_tensor, stochastic)
        return actions.cpu().numpy()[0], log_probs.cpu().numpy()[0], values.cpu().numpy()[0]

    def save(self, path: str):
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path: str):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
