python -m ttt_reward_models.cli_train_mnist_latent_matched_sft \
  --generator_ckpt outputs/mnist_gan/mnist_gan_final.pt \
  --manifest_path outputs/mnist_assigned_dataset/manifest.jsonl \
  --output_dir outputs/mnist_latent_matched_sft \
  --epochs 5 \
  --batch_size 64 \
  --lr 1e-4 \
  --preserve_weight 0.25
