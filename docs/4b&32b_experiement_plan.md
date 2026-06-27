7. 推荐模型/模式选择

主推荐：Qwen3-4B student + Qwen3-32B PI teacher，thinking-on

这是我最推荐的下一轮。

Student rollout: Qwen3-4B, thinking-on
External trace generator: Qwen3-32B, thinking-on
PI self-teacher scorer: Qwen3-4B base/no-adapter + PI, thinking-on
Eval: thinking-on

为什么：

1. 4B 比 1.7B 有更好的 support，RLSD 只在已有 support 内重分配 credit；
2. 32B trace 的质量显著高于 8B；
3. thinking-on 提供足够多的 reasoning tokens 让 RLSD 做 token-level credit；
4. 4B 仍然能在 4×H200 上 LoRA 快速训练。

不推荐继续作为主线：1.7B / 8B / thinking-off

可以作为补充 ablation，但不建议做主线。1.7B 太弱，8B trace 对 1.7B self-teacher 的帮助有限；thinking-off 又减少了 RLSD 的可用 token-level reasoning signal。

⸻

8. ExtPI-RLSD 推荐超参：Qwen3-4B / 32B / thinking-on

LoRA

rank = 32
alpha = 64
dropout = 0
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
bf16
gradient_checkpointing = true

不建议一开始 r64。先让训练稳定。

Rollout

train_batch_size = 8
rollout_n = 4
trajectories/update = 32
total_steps = 100
ppo_epochs = 1
temperature = 1.0
top_p = 1.0
max_prompt_length = 2048
max_response_length = 4096

thinking-on 下 2048 response 可能仍然偏短。你之前 2048 eval 里仍有截断；4B thinking-on 会更长。所以建议训练 max_response_length=4096。如果显存压力太大，先用 3072，不要回到 1024/2048。

RLSD

第一轮稳定版：

actor_lr = 5e-7
rlsd_lambda = 0.3
rlsd_reweight_clip_range = 0.2
rlsd_negative_only = true
rlsd_delta_student_logp_source = current
clip_ratio_low = 0.2
clip_ratio_high = 0.28
entropy_coeff = 0
use_kl_loss = false

Teacher scorer mode

teacher_update_mode = periodic_snapshot
teacher_sync_interval = 10

⸻

9. OPD baseline 推荐超参：Qwen3-4B / 32B / thinking-on

OPD 作为 baseline 应该更强，建议：

student = Qwen3-4B
teacher = Qwen3-32B
mode = thinking-on
train_batch_size = 8
rollout_n = 4
total_steps = 100
actor_lr = 3e-7 或 5e-7
max_response_length = 4096
opd_logratio_clip_abs = 2.0
teacher_pool: 1 GPU if possible
actor_pool: 3 GPUs

如果 OPD logratio 太强：

opd/logratio_abs_mean > 0.8
opd/logratio_p99 > 4
response length drift > 20%

就降到：

actor_lr = 3e-7
opd_logratio_clip_abs = 1.0

OPD 的理论价值在于 teacher 直接在 student trajectory 上给 token logprobs；Thinking Machines 的伪代码也是 student rollout、teacher compute logprobs、用 negative reverse KL 作为 per-token advantage。 
 

13. 推荐具体参数

GRPO

ACTOR_LR=5e-7
TRAIN_BATCH_SIZE=8
ROLLOUT_N=4
TOTAL_STEPS=100
MAX_RESPONSE_LENGTH=4096
MAX_PROMPT_LENGTH=2048
LoRA r32 alpha64

OPD

ACTOR_LR=3e-7 or 5e-7
OPD_LOGRATIO_CLIP_ABS=2.0
MAX_RESPONSE_LENGTH=4096
actor GPUs=3
teacher GPUs=1
teacher=Qwen3-32B

ExtPI-RLSD

ACTOR_LR=5e-7
RLSD_LAMBDA=0.3
RLSD_CLIP_RANGE=0.2
RLSD_NEGATIVE_ONLY=True
RLSD_DELTA_SOURCE=current
MAX_RESPONSE_LENGTH=4096
TEACHER_MAX_PROMPT_LENGTH=3072
online_teacher_thinking=True
student thinking=True
external trace generator thinking=True

实验数据
ExtPI-RLSD 训练集：
/data/users/rchen/extpi-rlsd/datasets/splits/frontier_mvp_qwen3_32b_thinking_on/mvp_train.parquet
OPD matched 训练集：
/data/users/rchen/extpi-rlsd/datasets/splits/frontier_mvp_opd_matched_qwen3_32b_thinking_on/mvp_train.parquet
同时各自有 64 条 smoke：
.../smoke.parquet