python -m ttt_reward_models.cli_build_mnist_assigned_noise_dataset \
  --generator_ckpt outputs/mnist_gan/mnist_gan_final.pt \
  --output_dir outputs/mnist_assigned_dataset \
  --data_root ./data \
  --max_items 256 \
  --assign_steps 200 \
  --assign_lr 5e-2
