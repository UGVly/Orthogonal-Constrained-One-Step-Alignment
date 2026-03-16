python -m ttt_reward_models.cli_train_mnist_gan \
  --output_dir outputs/mnist_gan \
  --data_root ./data \
  --epochs 10 \
  --batch_size 128 \
  --z_dim 64
