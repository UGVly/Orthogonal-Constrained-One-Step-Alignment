python -m ttt_reward_models.cli_em_cifar_latent_matched_sft \
  --generator_ckpt outputs/cifar_gan/cifar_gan_final.pt \
  --output_dir outputs/cifar_em_latent_matched_sft \
  --em_rounds 3 \
  --assign_max_items 512 \
  --assign_steps 300 \
  --sft_epochs 2
