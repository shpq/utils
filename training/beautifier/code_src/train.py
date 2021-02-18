import os
from tqdm import tqdm
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from code_src.model import EncoderDecoder
from code_src.model_res import ResModel
from code_src.datagenerator import DataGenerator
from code_src.augmentation import Augmentation
from code_src import beauty
from time import sleep
physical_devices = tf.config.list_physical_devices('GPU')

for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)


def train_model(cfg):
    global cross_entropy
    global Lambda
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,  # )
                                                       label_smoothing=cfg.training.label_smooth)
    Lambda = cfg.training.Lambda
    # print("sleeping")
    # sleep(60 * 60 * 2)
    augmentator = Augmentation(cfg)
    train_generator = DataGenerator(cfg, transform=augmentator)
    # test_generator = DataGenerator(cfg, transform=augmentator, mode="test")
    if cfg.model.name == "EncoderDecoder":
        model = EncoderDecoder(cfg)
    elif cfg.model.name == "ResModel":
        model = ResModel(cfg)
    else:
        raise NotImplementedError

    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.training.lr_beauty, beta_1=0.5)
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.training.lr, beta_1=0.5)
    # model.compile(optimizer=optimizer,
    #               loss="mean_absolute_error",
    #               metrics=["MeanAbsoluteError"])
    model.build((cfg.training.batch_size, *cfg.training.size, 3))

    model.summary()

    if cfg.model.pretrained_path is not None:
        model.load_weights(cfg.model.pretrained_path)
        print("------- WEIGHTS LOADED -------")

    filepath = os.path.join(
        cfg.training.checkpoints_path,
        '{epoch:02d}-loss-{val_loss:.2f}.h5'
    )

    beauty_model = beauty.load_model(cfg)

    os.makedirs(cfg.training.checkpoints_path, exist_ok=True)
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(patience=cfg.training.epoch_stop),
    #     tf.keras.callbacks.ModelCheckpoint(
    #         monitor="val_acc",
    #         filepath=filepath,),
    #     tf.keras.callbacks.ReduceLROnPlateau(
    #         patience=cfg.training.epoch_reduce,
    #         verbose=1,
    #     )
    # ]
    train(cfg, model, beauty_model, train_generator,
          generator_optimizer, discriminator_optimizer)
    # H = model.fit(
    #     train_generator,
    #     validation_data=test_generator,
    #     # steps_per_epoch=len(train_generator),
    #     # validation_steps=len(test_generator),
    #     epochs=cfg.training.epochs, callbacks=callbacks,
    # )


def generator_loss(fake_output, img_1, img_2, img_2_changed):
    # size_dest = (152, 152)
    # img_1, img_2 = tf.image.resize(img_1, size_dest), tf.image.resize(img_2, size_dest)
    l1 = tf.reduce_mean(
        tf.keras.losses.mean_absolute_error(img_2, img_2_changed))
    # l1 =  4e-7 * (tf.reduce_sum(tf.abs(img_1[:, :, :-1, :] - img_1[:, :, 1:, :])) +
    # tf.reduce_sum(tf.abs(img_1[:, -1:, :, :] - img_1[:, 1:, :, :])))
    ce = cross_entropy(tf.ones_like(fake_output), fake_output)
    loss = ce + tf.cast(Lambda * l1, tf.float32)
    return loss, l1, ce

# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# def calc_reg_loss(images_beautified):
#     out = tf.reduce_sum(tf.abs(images_beautified[:, :, :-1, :] - images_beautified[:, :, 1:, :])) +
#          tf.reduce_sum(tf.abs(images_beautified[:, -1:, :, :] - images_beautified[:, 1:, :, :]))

#     return out


def train(cfg, generator, discriminator, train_generator, generator_optimizer, discriminator_optimizer):
    @tf.function
    def train_step(images_ugly, images_beauty):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            images_beautified = generator(images_ugly, training=True)
            images_beauty_changed = generator(images_beauty, training=True)

            real_output = discriminator(images_beauty, training=True)
            fake_output = discriminator(images_beautified, training=True)

            gen_loss, l1, ce = generator_loss(
                fake_output, images_beautified, images_beauty, images_beauty_changed)
            # gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss, l1, ce
        # return gen_loss, disc_loss
    total = len(train_generator)
    N_check = cfg.training.checkpoint_every
    # checkpoint_every = [ind for ind in range(total) if (ind % (total // 100) == 0 and ind != 0) or ind == total - 1]
    for epoch in range(cfg.training.epochs):
        checkpoint_every = [ind for ind in range(total) if (
            ind % (total // N_check) == 0 and ind != 0) or ind == total - 1]
        pbar = tqdm(enumerate(train_generator),
                    total=total)
        gen_loss_sum = 0
        disc_loss_sum = 0
        l1_loss_sum = 0
        ind_scale_value = 0
        for ind, chunk in pbar:
            scale_value = 1 / max(1, ind_scale_value)
            images_ugly, images_beauty = chunk
            gen_loss, disc_loss, l1, ce = train_step(
                images_ugly, images_beauty)
            # gen_loss, disc_loss = train_step(images_ugly, images_beauty)
            gen_loss_sum += gen_loss
            disc_loss_sum += disc_loss
            l1_loss_sum += l1
            pbar.set_description(
                "Epoch: {} -check at {}- gen_loss: {:.4f} ({:.4f}), disc_loss: {:.4f} ({:.4f}), l1_loss: {:.4f} ({:.4f})".format(
                    epoch,
                    checkpoint_every[0],
                    gen_loss_sum * scale_value,
                    gen_loss,
                    disc_loss_sum * scale_value,
                    disc_loss,
                    l1_loss_sum * scale_value,
                    l1
                )
            )
            # pbar.set_description(
            #     "Epoch: {} gen_loss: {:.4f}, disc_loss: {:.4f}".format(
            #         epoch,
            #         gen_loss_sum * scale_value,
            #         disc_loss_sum * scale_value
            #     )
            # )
            ind_scale_value += 1
            if ind in checkpoint_every:
                checkpoint_every.pop(0)
                filepath_gen = os.path.join(
                    cfg.training.checkpoints_path,
                    f'gen_ep_{round(epoch + ind / total, 2)}_gen_{round(float(gen_loss_sum* scale_value), 2)}_disc_{round(float(disc_loss_sum* scale_value), 2)}_reg_{round(float(l1_loss_sum* scale_value), 2)}_model.h5'
                )

                filepath_disc = os.path.join(
                    cfg.training.checkpoints_path,
                    f'disc_ep_{round(epoch + ind / total, 2)}_gen_{round(float(gen_loss_sum* scale_value), 2)}_disc_{round(float(disc_loss_sum* scale_value), 2)}_reg_{round(float(l1_loss_sum* scale_value), 2)}_model.h5'
                )
                gen_loss_sum = 0
                disc_loss_sum = 0
                l1_loss_sum = 0
                ind_scale_value = 0
                generator.save_weights(filepath_gen)
                discriminator.save_weights(filepath_disc)
