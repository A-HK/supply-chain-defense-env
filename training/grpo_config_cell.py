from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from training.callbacks import MetricsCallback
from training.trajectory_filter import CollapseDetector
import gc

gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ''
use_bf16 = any(x in gpu_name for x in ['a100','a10','h100','l4','l40'])

# NOTE: max_prompt_length was removed in TRL 1.x (Unsloth patches against 1.x)
# Use max_completion_length to control generation length instead
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    temperature=TEMPERATURE,
    beta=KL_COEFF,
    scale_rewards=False,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    bf16=use_bf16,
    fp16=not use_bf16,
    logging_steps=1,
    logging_first_step=True,
    log_completions=True,
    disable_tqdm=True,
    save_strategy='steps',
    save_steps=20,
    save_total_limit=3,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy='every_save',
    warmup_ratio=0.1,
    reward_weights=[1.0, 0.3, 0.2],
    report_to=[],
)

collapse_detector = CollapseDetector(window=15)

class CurriculumGRPOCallback(TrainerCallback):
    def __init__(self):
        self.metrics_cb = MetricsCallback(str(ARTIFACT_DIR / 'grpo_metrics.jsonl'))
        self.episode_rewards = []; self.losses = []; self.step_count = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        global CURRENT_DIFFICULTY
        if not logs: return
        loss = logs.get('loss') or logs.get('train_loss')
        if loss is not None: self.losses.append(loss)
        reward = logs.get('reward')
        if reward is not None:
            self.step_count += 1
            self.episode_rewards.append(reward)
            self.metrics_cb.log({'type':'grpo_step','step':self.step_count,'reward':reward,'loss':loss,'difficulty':CURRENT_DIFFICULTY})
            alert = collapse_detector.update(reward=reward)
            if alert: print(f'  WARNING: {alert}')
            ATTACKER.observe_defender({'benchmark_score': reward/2.0, 'breakdown':{'notify_ratio':0.5,'rotate_ratio':0.5}})
            if len(self.episode_rewards) >= ROLLING_WINDOW:
                avg = np.mean(self.episode_rewards[-ROLLING_WINDOW:])
                if CURRENT_DIFFICULTY == 'easy' and avg > EASY_ADVANCE * 2:
                    CURRENT_DIFFICULTY = 'medium'; print(f'\nCURRICULUM -> medium (avg={avg:.3f})')
                elif CURRENT_DIFFICULTY == 'medium' and avg > MEDIUM_ADVANCE * 2:
                    CURRENT_DIFFICULTY = 'hard'; print(f'\nCURRICULUM -> hard (avg={avg:.3f})')
            if self.step_count % 5 == 0:
                recent = self.episode_rewards[-5:]
                adv = ATTACKER.get_metrics()
                print(f'  step {self.step_count}: reward={reward:.3f} avg5={np.mean(recent):.3f} diff={CURRENT_DIFFICULTY} adv={adv["adversarial_level"]:.2f}')

curriculum_cb = CurriculumGRPOCallback()
print(f'Config ready')