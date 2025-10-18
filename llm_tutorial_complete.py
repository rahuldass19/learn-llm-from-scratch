"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    BUILD & TRAIN A 100M LLM - COMPLETE TUTORIAL          ║
║                         From Architecture to Generation                  ║
║                        Works on Google Colab FREE Tier                   ║
║                                                                          ║
║  Target: 100M+ parameters (GPT-2 Small size)                              ║
║  GPU: T4 (15GB) - Colab Free Tier                                       ║
║  Time: 5-7 hours total (building + training)                             ║
║  Dataset: WikiText-2 (~14K samples)                                      ║
║  Cost: $0 (completely free)                                              ║
║                                                                          ║
║  What you'll learn:                                                      ║
║  • Transformer architecture from scratch                                 ║
║  • Rotary positional embeddings (RoPE)                                   ║
║  • Flash Attention mechanism                                             ║
║  • Complete training pipeline                                            ║
║  • Text generation                                                       ║
║                                                                          ║
║  Created for educational purposes - learn by building!                   ║
║  Contact: rahuldass1901@gmail.com                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

⚠️  CRITICAL: This is NOT a copy-paste tutorial!
    You will BUILD each component step by step.
    Use Claude/ChatGPT to help fill in the code.
    
    The structure and guidance are here.
    The implementation is YOUR job.
    
    That's how you actually learn and OWN the code. 🔥

📋 BEFORE STARTING:
   1. Open this in Google Colab
   2. Runtime → Change runtime type → GPU (T4)
   3. Have Claude or ChatGPT ready in another tab
   4. Patience! This takes time but you'll learn REAL skills

💡 WHAT TO EXPECT AFTER TRAINING:
   Your model will:
   ✅ Generate grammatically correct English
   ✅ Use proper punctuation and structure
   ✅ Show it learned language patterns
   ⚠️  NOT be factually accurate (needs 100x more data)
   ⚠️  NOT be ChatGPT-level (needs 1000x more training)
   
   BUT YOU'LL UNDERSTAND:
   ✅ How transformers work internally
   ✅ How training reduces loss
   ✅ How text generation works
   ✅ The path to scale from 100M → 7B → 175B
   
   This is EDUCATION, not production. You're learning the fundamentals
   that OpenAI, Meta, and Google use at massive scale.
"""

# ============================================================================
# PART 1: ENVIRONMENT SETUP (PROVIDED)
# ============================================================================
"""
First, let's set up the environment and check GPU availability.
"""

# Install required packages
!pip install torch torchvision torchaudio --quiet
!pip install transformers datasets --quiet

# Imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import logging
import time
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Check environment
print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {total_memory:.2f} GB")
    
    if total_memory < 14:
        print("⚠️  WARNING: Less than 14GB GPU memory detected.")
        print("   You may need to reduce batch size.")
else:
    print("❌ ERROR: No GPU detected!")
    print("   Go to: Runtime → Change runtime type → GPU")
    print("   Then restart this notebook.")

print("=" * 70)

"""
✅ VALIDATION:
   - PyTorch 2.0+
   - CUDA available: True
   - GPU: Tesla T4 (or similar)
   - Memory: 14-16GB
   
If all checks pass, continue. Otherwise fix setup first.
"""


# ============================================================================
# PART 2: MODEL CONFIGURATION
# ============================================================================
"""
INSTRUCTION: Define model architecture for 100M parameters
DIFFICULTY: ⭐⭐ (Medium)

📚 CONTEXT:
This configuration defines a 124M parameter model (same size as GPT-2 Small).
The architecture uses:
- 768 hidden dimensions
- 12 layers, 12 attention heads
- 3072 FFN intermediate size (4x hidden)
- 50,257 vocabulary (GPT-2 tokenizer)
- 512 context length
"""

class ModelConfig:
    def __init__(self):
        # Architecture
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.vocab_size = 50257  # GPT-2 tokenizer
        self.max_sequence_length = 512
        
        # Regularization
        self.hidden_dropout_prob = 0.1
        self.attention_dropout_prob = 0.1
        self.layer_norm_epsilon = 1e-5
        
        # Training (will be used later)
        self.learning_rate = 3e-4
        self.weight_decay = 0.1
        self.max_grad_norm = 1.0
        self.gradient_accumulation_steps = 8
        
        # Optimization flags
        self.use_flash_attention = True
        self.gradient_checkpointing = True
        self.tie_word_embeddings = True
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
        
        # Batch sizing for T4 GPU (memory optimized)
        self.batch_size = 2
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
    
    def calculate_parameters(self):
        """Calculate total model parameters"""
        embedding_params = self.vocab_size * self.hidden_size
        
        attention_params = (
            4 * self.hidden_size * self.hidden_size +
            2 * self.hidden_size
        )
        ffn_params = (
            2 * self.hidden_size * self.intermediate_size +
            self.hidden_size + self.intermediate_size
        )
        block_params = (attention_params + ffn_params) * self.num_hidden_layers
        
        output_params = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        
        total = embedding_params + block_params + output_params
        return total
    
    def print_config(self):
        """Print configuration summary"""
        total_params = self.calculate_parameters()
        print('\n' + '='*70)
        print('MODEL CONFIGURATION - 100M')
        print('='*70)
        print(f'Architecture: {self.num_hidden_layers} layers')
        print(f'Hidden size: {self.hidden_size}')
        print(f'Attention heads: {self.num_attention_heads}')
        print(f'Vocabulary: {self.vocab_size:,} tokens')
        print(f'Context length: {self.max_sequence_length} tokens')
        print(f'Total parameters: {total_params:,} ({total_params/1e6:.1f}M)')
        print(f'Batch size: {self.batch_size} (effective: {self.effective_batch_size})')
        print(f'Device: {self.device}')
        print('='*70 + '\n')

# Test your config
config = ModelConfig()
config.print_config()


# ============================================================================
# PART 3: ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ============================================================================
"""
INSTRUCTION: Implement RoPE for position encoding
DIFFICULTY: ⭐⭐⭐ (Hard)

📚 CONTEXT:
RoPE encodes position information by rotating query/key vectors.
Used in modern LLMs (LLaMA, GPT-NeoX) - better than fixed embeddings.
"""

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)
    
    @staticmethod
    def rotate_half(x):
        """Rotate half the hidden dims of the input"""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

# Test RoPE
rope = RotaryPositionalEmbedding(dim=64)
cos, sin = rope(seq_len=10, device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"RoPE test - Cos shape: {cos.shape}, Sin shape: {sin.shape}")


# ============================================================================
# PART 4: FLASH ATTENTION
# ============================================================================
"""
INSTRUCTION: Implement multi-head attention with RoPE
DIFFICULTY: ⭐⭐⭐⭐ (Very Hard) - This is the CORE of transformers!
"""

class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.attention_dropout_prob
        
        assert self.hidden_size % self.num_heads == 0
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config.max_sequence_length)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        cos, sin = self.rotary_emb(seq_length, hidden_states.device)
        cos = cos.view(1, seq_length, 1, self.head_dim)
        sin = sin.view(1, seq_length, 1, self.head_dim)
        
        query_rot = self.rotary_emb.rotate_half(query)
        query = query * cos + query_rot * sin
        
        key_rot = self.rotary_emb.rotate_half(key)
        key = key * cos + key_rot * sin
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

# Test Attention
config_test = ModelConfig()
attn = FlashAttention(config_test)
if torch.cuda.is_available():
    attn = attn.cuda()
    x = torch.randn(2, 10, 768).cuda()
    out = attn(x)
    print(f"Attention test PASSED - Input: {x.shape}, Output: {out.shape}")


# ============================================================================
# PART 5: TRANSFORMER BLOCK
# ============================================================================
"""
INSTRUCTION: Combine attention and feed-forward into transformer block
DIFFICULTY: ⭐⭐⭐ (Hard)
"""

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.attention = FlashAttention(config)
        self.ffn_up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn_up(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.ffn_down(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# Test TransformerBlock
block = TransformerBlock(config_test)
if torch.cuda.is_available():
    block = block.cuda()
    x = torch.randn(2, 10, 768).cuda()
    out = block(x)
    print(f"TransformerBlock test PASSED - Input: {x.shape}, Output: {out.shape}")


# ============================================================================
# PART 6: COMPLETE MODEL
# ============================================================================
"""
INSTRUCTION: Assemble complete language model
DIFFICULTY: ⭐⭐⭐⭐ (Very Hard)
"""

class NanoLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.output.weight = self.embeddings.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.embeddings(input_ids)
        
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits}

# Create and test model
print("\n" + "="*70)
print("CREATING MODEL")
print("="*70)

config = ModelConfig()
model = NanoLM(config)

if torch.cuda.is_available():
    model = model.cuda()
    print(f"Model moved to GPU")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1e6:.0f}MB (fp32)")
print("="*70 + "\n")

if torch.cuda.is_available():
    test_input = torch.randint(0, 50257, (2, 10)).cuda()
    test_output = model(test_input)
    print(f"Forward pass test - Logits shape: {test_output['logits'].shape}")

print("\nMODEL CREATED SUCCESSFULLY!")


# ============================================================================
# PART 7: DOWNLOAD WIKITEXT-2 & SETUP TOKENIZER
# ============================================================================
"""
Now we'll download the training data and set up the tokenizer.
We use GPT-2 tokenizer which has vocab size of 50,257 (matches our model).
"""

from datasets import load_dataset
from transformers import GPT2TokenizerFast

print("\n" + "="*70)
print("DOWNLOADING WIKITEXT-2 & LOADING TOKENIZER")
print("="*70)

# Download WikiText-2
print("\nDownloading WikiText-2...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(f"Dataset loaded: {len(dataset['train'])} train samples")

# Load GPT-2 tokenizer
print("\nLoading GPT-2 tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer loaded")
print(f"  Vocab size: {tokenizer.vocab_size:,}")
print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# Verify vocab size matches model
assert tokenizer.vocab_size == config.vocab_size
print(f"\nVocab size matches model: {config.vocab_size:,}")
print("="*70)


# ============================================================================
# PART 8: CREATE TRAINING DATASET
# ============================================================================
"""
INSTRUCTION: Create PyTorch Dataset for training
DIFFICULTY: ⭐⭐⭐ (Hard)
"""

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class WikiTextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, split='train', max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        print(f"\n{'='*70}")
        print(f"TOKENIZING {split.upper()} SPLIT")
        print(f"{'='*70}")
        
        self.samples = []
        raw_data = hf_dataset[split]
        
        print(f"Tokenizing {len(raw_data)} samples...")
        for idx in tqdm(range(len(raw_data)), desc=f"Tokenizing {split}"):
            text = raw_data[idx]['text']
            
            if len(text.strip()) < 10:
                continue
            
            encoded = tokenizer.encode(text, max_length=max_length, truncation=True)
            
            if len(encoded) >= 50:
                self.samples.append(encoded)
        
        print(f"\nTokenized {len(self.samples)} valid samples")
        print(f"  (Filtered out {len(raw_data) - len(self.samples)} empty/short samples)")
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        encoded = self.samples[idx]
        
        padding_length = self.max_length - len(encoded)
        input_ids = encoded + [self.tokenizer.pad_token_id] * padding_length
        attention_mask = [1] * len(encoded) + [0] * padding_length
        
        labels = input_ids.copy()
        labels = [-100 if mask == 0 else token for token, mask in zip(labels, attention_mask)]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Create datasets
print("\n" + "="*70)
print("CREATING DATASETS")
print("="*70)

train_dataset = WikiTextDataset(dataset, tokenizer, split='train', max_length=512)
val_dataset = WikiTextDataset(dataset, tokenizer, split='validation', max_length=512)

# Create dataloaders
batch_size = 2
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("\n" + "="*70)
print("DATALOADER SUMMARY")
print("="*70)
print(f"Training samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print(f"Batch size: {batch_size}")
print(f"Training batches: {len(train_loader):,}")
print(f"Validation batches: {len(val_loader):,}")
print("="*70)

test_batch = next(iter(train_loader))
print(f"\nTest batch:")
print(f"  input_ids: {test_batch['input_ids'].shape}")
print(f"  attention_mask: {test_batch['attention_mask'].shape}")
print(f"  labels: {test_batch['labels'].shape}")

print("\nDatasets ready for training!")


# ============================================================================
# PART 9: TRAINING CONFIGURATION
# ============================================================================
"""
INSTRUCTION: Set up training configuration
DIFFICULTY: ⭐⭐ (Medium)
"""

class TrainingConfig:
    def __init__(self):
        self.learning_rate = 3e-4
        self.min_learning_rate = 3e-5
        self.warmup_steps = 500
        self.num_epochs = 3
        self.gradient_accumulation_steps = 8
        self.max_grad_norm = 1.0
        self.save_every = 1000
        self.log_every = 10
        self.eval_every = 500
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_total_steps(self, train_loader):
        steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
        return steps_per_epoch * self.num_epochs

train_config = TrainingConfig()
total_steps = train_config.get_total_steps(train_loader)

print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"Learning rate: {train_config.learning_rate}")
print(f"Warmup steps: {train_config.warmup_steps}")
print(f"Gradient accumulation: {train_config.gradient_accumulation_steps}")
print(f"Effective batch size: {batch_size * train_config.gradient_accumulation_steps}")
print(f"Epochs: {train_config.num_epochs}")
print(f"Total steps: {total_steps}")
print(f"Estimated time: ~1.5-2 hours")
print("="*70)

# Create optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Create LR scheduler
def get_lr_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(
    optimizer,
    warmup_steps=train_config.warmup_steps,
    total_steps=total_steps
)

print(f"\nOptimizer: AdamW")
print(f"Scheduler: Warmup + Cosine Decay")
print(f"\nTraining setup complete!")


# ============================================================================
# PART 10: TRAINING LOOP
# ============================================================================
"""
INSTRUCTION: Implement the training loop
DIFFICULTY: ⭐⭐⭐⭐ (Very Hard)

This is where your model actually learns!
"""

os.makedirs('checkpoints', exist_ok=True)

def train_step(model, batch, optimizer, scheduler, config, step_in_accum):
    """Single training step with gradient accumulation"""
    input_ids = batch['input_ids'].to(config.device)
    attention_mask = batch['attention_mask'].to(config.device)
    labels = batch['labels'].to(config.device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs['loss']
    
    loss = loss / config.gradient_accumulation_steps
    loss.backward()
    
    if (step_in_accum + 1) % config.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return loss.item() * config.gradient_accumulation_steps

def evaluate(model, val_loader, config, max_batches=100):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs['loss'].item()
            num_batches += 1
    
    model.train()
    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity

# Training loop
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Device: {train_config.device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("="*70)

model.train()
optimizer.zero_grad()

global_step = 0
step_in_accum = 0
best_val_loss = float('inf')
training_losses = []
start_time = time.time()

print("\nTraining started! Watch the loss decrease...\n")

try:
    for epoch in range(train_config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{train_config.num_epochs}")
        print(f"{'='*70}\n")
        
        for batch_idx, batch in enumerate(train_loader):
            loss_value = train_step(model, batch, optimizer, scheduler, train_config, step_in_accum)
            training_losses.append(loss_value)
            step_in_accum += 1
            
            if step_in_accum % train_config.gradient_accumulation_steps == 0:
                global_step += 1
                step_in_accum = 0
                
                if global_step % train_config.log_every == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9
                    avg_loss = sum(training_losses[-train_config.log_every:]) / len(training_losses[-train_config.log_every:])
                    
                    print(
                        f"Step {global_step:4d}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed/60:.1f}m | "
                        f"GPU: {mem_gb:.1f}GB"
                    )
                
                if global_step % train_config.eval_every == 0:
                    print(f"\n--- Evaluating at step {global_step} ---")
                    val_loss, val_ppl = evaluate(model, val_loader, train_config)
                    print(f"Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        print(f"New best validation loss!")
                    print()
                
                if global_step % train_config.save_every == 0:
                    checkpoint_path = f'checkpoints/checkpoint_step_{global_step}.pt'
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}\n")
                
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
            
            if global_step >= total_steps:
                break
        
        if global_step >= total_steps:
            break

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user!")

# Final evaluation
print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
total_time = time.time() - start_time
print(f"Total time: {total_time/3600:.2f} hours")
print(f"Total steps: {global_step}")

print(f"\nFinal evaluation...")
final_val_loss, final_val_ppl = evaluate(model, val_loader, train_config)
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Perplexity: {final_val_ppl:.2f}")

# Save final model
final_checkpoint = 'checkpoints/final_model.pt'
torch.save({
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': final_val_loss,
}, final_checkpoint)
print(f"\nFinal model saved: {final_checkpoint}")

print("\n" + "="*70)
print("YOUR MODEL IS TRAINED!")
print("="*70)


# ============================================================================
# PART 11: TEXT GENERATION
# ============================================================================
"""
INSTRUCTION: Generate text from your trained model
DIFFICULTY: ⭐⭐⭐ (Hard)

Now let's see what your model can do!
"""

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_p=0.9):
    """Generate text from a prompt"""
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.config.device)
    
    print(f"Prompt: {prompt}")
    print(f"\nGenerating {max_length} tokens...\n")
    print("="*70)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs['logits']
            
            next_token_logits = logits[0, -1, :] / temperature
            
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(generated_text)
    print("="*70)
    return generated_text

# Test your trained model
print("\n" + "="*70)
print("TESTING YOUR TRAINED MODEL")
print("="*70)

prompts = [
    "The history of artificial intelligence",
    "In the field of computer science,",
    "The theory of relativity states that",
    "During World War II,",
    "The human brain is"
]

print("\nGenerating text from 5 different prompts...\n")

for i, prompt in enumerate(prompts, 1):
    print(f"\n{'='*70}")
    print(f"GENERATION {i}/5")
    print(f"{'='*70}\n")
    generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8)
    print("\n")

print("\n" + "="*70)
print("ALL GENERATIONS COMPLETE!")
print("="*70)


# ============================================================================
# PART 12: WHAT YOU LEARNED
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                         CONGRATULATIONS! 🎉                              ║
╚══════════════════════════════════════════════════════════════════════════╝

YOU JUST BUILT AND TRAINED A 124M PARAMETER LANGUAGE MODEL!

✅ WHAT YOU ACCOMPLISHED:
   • Built a transformer architecture from scratch
   • Implemented RoPE (Rotary Positional Embeddings)
   • Created Flash Attention mechanism
   • Built complete training pipeline
   • Trained on WikiText-2 dataset
   • Generated text from your model

📊 YOUR MODEL:
   • Parameters: ~124M (same size as GPT-2 Small)
   • Architecture: 12 layers, 12 attention heads
   • Technology: 2024-2025 state-of-the-art (RoPE, Flash Attention)
   • Training: ~1.5-2 hours on free GPU
   • Cost: $0

🎓 WHAT YOU NOW UNDERSTAND:
   ✓ How transformers work internally
   ✓ How attention mechanisms process sequences
   ✓ How models learn through gradient descent
   ✓ How loss decreases during training
   ✓ How text generation works
   ✓ The path from 100M → 1B → 7B → 175B parameters

💡 ABOUT YOUR MODEL'S GENERATIONS:
   Your model generates grammatically correct text but may not be
   factually accurate. This is NORMAL and EXPECTED because:
   
   • Small dataset (14K samples vs millions for GPT-2)
   • Limited training (3 epochs vs weeks/months)
   • Educational model (learning fundamentals, not production)
   
   What matters: You understand HOW it works!

🚀 WHAT'S NEXT?

IMMEDIATE IMPROVEMENTS:
   1. Train longer (10-20 epochs instead of 3)
   2. Use larger dataset (WikiText-103, 100x larger)
   3. Increase model size (350M, 1B parameters)
   4. Fine-tune on specific tasks

ADVANCED TOPICS:
   1. Instruction tuning (make it follow commands)
   2. RLHF (Reinforcement Learning from Human Feedback)
   3. Mixture of Experts (MoE)
   4. Multi-GPU training (scale to 7B+)

💪 YOU'RE DIFFERENT NOW:
   Most people: Use APIs, don't understand internals
   You: Built it from scratch, own the knowledge
   
   Most portfolios: Copy-paste tutorials
   Yours: Original implementation you can explain

🌟 WHAT YOU CAN SAY NOW:
   "I built a 124M parameter transformer model from scratch,
   implemented modern techniques like RoPE and Flash Attention,
   and trained it to generate text. I understand how ChatGPT
   works internally, not just how to use it."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    YOU DIDN'T JUST LEARN - YOU BUILT.
                         THAT'S WHAT MAKES YOU DIFFERENT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📧 Questions? Want to connect?
   Email: rahuldass1901@gmail.com
   
   Share your success! Let others know you built this.
   The more people who understand AI deeply, the better.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    NOW GO BUILD SOMETHING AMAZING.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


# ============================================================================
# QUICK REFERENCE: USEFUL COMMANDS
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                          QUICK REFERENCE                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

📝 GENERATE MORE TEXT:
   generate_text(model, tokenizer, "Your prompt here", max_length=100)

💾 SAVE YOUR MODEL:
   torch.save(model.state_dict(), 'my_model.pt')

📂 LOAD YOUR MODEL:
   model.load_state_dict(torch.load('my_model.pt'))
   model.eval()

🔍 CHECK MODEL SIZE:
   total_params = sum(p.numel() for p in model.parameters())
   print(f"Parameters: {total_params/1e6:.1f}M")

💻 CHECK GPU MEMORY:
   print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

🧹 CLEAR GPU MEMORY:
   torch.cuda.empty_cache()
   gc.collect()

📊 CALCULATE PERPLEXITY:
   perplexity = math.exp(loss_value)

🎲 EXPERIMENT WITH GENERATION:
   # More creative (higher temperature)
   generate_text(model, tokenizer, prompt, temperature=1.2)
   
   # More focused (lower temperature)
   generate_text(model, tokenizer, prompt, temperature=0.5)
   
   # More diverse (higher top_p)
   generate_text(model, tokenizer, prompt, top_p=0.95)

══════════════════════════════════════════════════════════════════════════

Tutorial Complete! You now understand LLMs at a fundamental level.

Keep experimenting, keep learning, keep building! 🚀

══════════════════════════════════════════════════════════════════════════
""")
