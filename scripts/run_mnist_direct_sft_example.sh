python -m ttt_reward_models.cli_train_mnist_direct_sft \
  --generator_ckpt outputs/mnist_gan/mnist_gan_final.pt \
  --data_root ./data \
  --output_dir outputs/mnist_direct_sft \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --train_limit 2048
