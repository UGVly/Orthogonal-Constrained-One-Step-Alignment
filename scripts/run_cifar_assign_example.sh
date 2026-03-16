python -m ttt_reward_models.cli_build_cifar_assigned_noise_dataset \
  --generator_ckpt outputs/cifar_gan/cifar_gan_final.pt \
  --output_dir outputs/cifar_assigned_dataset \
  --max_items 512 \
  --assign_steps 300
