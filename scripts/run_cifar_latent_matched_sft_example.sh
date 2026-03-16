python -m ttt_reward_models.cli_train_cifar_latent_matched_sft \
  --generator_ckpt outputs/cifar_gan/cifar_gan_final.pt \
  --manifest_path outputs/cifar_assigned_dataset/manifest.jsonl \
  --output_dir outputs/cifar_latent_matched_sft \
  --epochs 5 \
  --batch_size 64 \
  --preserve_weight 0.25
