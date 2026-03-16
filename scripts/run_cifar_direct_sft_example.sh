python -m ttt_reward_models.cli_train_cifar_direct_sft \
  --generator_ckpt outputs/cifar_gan/cifar_gan_final.pt \
  --output_dir outputs/cifar_direct_sft \
  --epochs 5 \
  --batch_size 64
