class training:
    epoch = 100
    seed = 3035

    class optimizer:
        lr = 1e-5
        weight_decay = 1e-4

    class scheduler:
        lr_decay_start = -1
        lr_decay_frequency = 1  # decay frequency
        lr_decay_rate = 0.995  # when training epoch is more than it, the learning rate will be begun to decay


class dataset:
    path = '/root/Desktop/aaaa/datasets/ProcessedData/shanghaitech_part_B'
    mean_std = ([0.452016860247, 0.447249650955, 0.431981861591], [0.23242045939, 0.224925786257, 0.221840232611])
    scale = 100.
    train_batch_size = 16
    val_batch_size = 16
    num_workers = 12


class predict:
    model_path = "H:\\CrowdCount\\model\\res\\all_ep_64_mae_8.5_mse_14.1.pth"
