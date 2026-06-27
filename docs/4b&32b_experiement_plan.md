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

⸻

10. 是否更换数据集？

建议换，或者至少扩。

不建议继续只用 OPSD 同源 800 题

现在 800 条 frontier 对“快速验证”够，但对 4B/32B + thinking-on 的主实验偏小。Thinking Machines 的 reasoning OPD 使用的是 OpenThoughts-3 / Qwen3 reasoning 初始化相关数据；他们强调先有 SFT/mid-training support，再做 OPD。 

推荐数据策略

分两层：

层 1：继续保留 OPSD-compatible frontier

用于和前面实验可比：

800 train
512 matched_dev

但必须重新筛 Qwen3-4B thinking-on frontier。

层 2：新增更适合 Qwen3 reasoning 的 math frontier

推荐从这些来源构造候选池：

OpenThoughts / OpenThoughts3 style prompts
NuminaMath
DeepMath / DeepScaleR style math
MATH / OlympiadBench / AMC/AIME-like train split

筛选后只保留 Qwen3-4B 的 frontier，不要直接用全量。

目标：

train = 2000-5000 prompts
matched_dev = 512-1000 prompts

你如果坚持 batch size 8 和 100 step，那只会用 800 prompts。更大的数据集可以用于：

1. 构造更好的 matched_dev；
2. 挑更干净的 800 prompt；
3. 做 second run。

数据筛选比数据来源更重要

不要用太难题。RLSD/OPD 都依赖 student support。你之前文档里也写过，不要用 pS=0 的题做主训练集，保留 pS=0.25/0.5 这类 frontier。 

⸻

11. 是否要 thinking-on？

我的建议：

主线：thinking-on
消融：thinking-off

为什么主线 thinking-on

1. RLSD 做 token-level credit assignment，需要显式 reasoning tokens；
2. external 32B trace 本身是 reasoning trace，thinking-off 会把它压成“答案提示/风格提示”；
3. OPD/RLSD 都更适合长链路、多步推理任务；
4. 原 OPD reasoning 设定中 student/teacher 都是在 math reasoning 轨迹上做 dense supervision。 

需要注意

thinking-on 会增加长度和截断风险，所以要同步改：

max_response_length = 4096
eval max_new_tokens = 4096 or 6144
teacher PI trace max tokens = 2048-3072

如果 4096 训练太慢，可以：

先只筛 4096 内能完成的题
或用 concise trace / compressed PI

不要让大量训练 rollout 被截断。

⸻

12. 我建议的下一轮实验矩阵

在 4×H200 下，先跑这 4 个：

Run	Student	Teacher/PI	Thinking	目的
GRPO	Qwen3-4B	none	on	RL baseline
OPD	Qwen3-4B	Qwen3-32B logits	on	strong dense baseline
ExtPI-RLSD	Qwen3-4B	32B trace → 4B PI self-teacher	on	main
Shuffled-PI RLSD	Qwen3-4B	shuffled 32B trace	on	排除 prompt/trace style effect

必须先跑 5-step smoke，再跑 25-step，再决定是否跑满 100。

⸻

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

如果 25-step 后仍然太弱

只改一个变量：

RLSD_LAMBDA=0.5

不要同时改 LR、数据、thinking、lambda，否则无法归因。

⸻

14. 加一个我认为非常关键的新变体：contrastive ExtPI-RLSD

鉴于 RLCSD 指出 privileged context 会带来 style drift，我建议加一个轻量变体：

delta_correct = logp(Qwen + correct PI) - logp(Qwen no PI)
delta_wrong   = logp(Qwen + shuffled PI) - logp(Qwen no PI)
delta_final   = delta_correct - beta * delta_wrong

先设：

beta = 0.5

如果 delta_wrong 很大，说明 PI prompt 本身在改变风格，而不是题目相关语义。这个变体能让 RLSD 更聚焦 task-bearing tokens。

这和 RLSD 的精神一致：仍然由环境 reward 决定方向，只是把 privileged teacher 的 magnitude signal 去掉一部分非题目相关 style shift。RLSD 原文强调 teacher signal 只应该调节 update magnitude，而不是接管方向。 

⸻

15. 什么时候不该继续 RLSD，而应换任务/数据

如果你换成 Qwen3-4B/32B thinking-on 后仍出现：

GRPO >= ExtPI-RLSD ≈ shuffled-PI
OPD > ExtPI-RLSD

那说明问题不是模型规模，而是 ExtPI 的 semantic bridge 仍然不够强。

这时建议换数据，而不是继续调 RLSD：

1. 换到更自然有 privileged trace 的任务；
2. 换到有 verified intermediate steps 的 math/code；
3. 用 Qwen3-32B 生成更短、更结构化的 PI，不要长 CoT；
4. 做 contrastive PI。

如果出现：

OPD 也不涨

那说明 student support / 数据 frontier 有问题，应该重新筛数据或做 SFT cold start。

⸻

16. 我的最终建议

下一轮我会这样排：

第一优先级

Qwen3-4B student
Qwen3-32B external PI generator
thinking-on
重新筛 Qwen3-4B frontier
max_response_length=4096
ExtPI-RLSD: lr=5e-7, lambda=0.3, negative_only=True
OPD: lr=3e-7~5e-7, clip_abs=2.0

第二优先级

加入：

shuffled PI control
contrastive ExtPI-RLSD

第三优先级

如果 4B 正向，再扩：

Qwen3-8B student / Qwen3-32B teacher

不要现在直接跳 8B，除非你愿意减少实验矩阵。

⸻

一句话结论

你现在 RLSD 不明显，最可能是 1.7B + thinking-off + weak self-teacher bridge 的问题，不是 RLSD 公式本身的问题。
下一步应该换成 Qwen3-4B student + Qwen3-32B teacher/PI，全部 thinking-on，重新筛 Qwen3-4B recipient frontier，并把 max_response_length 提到 4096。在 4×H200 下，这比继续调 1.7B 或 thinking-off 更有希望看到清晰的 ExtPI-RLSD 信号。