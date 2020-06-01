from argparse import ArgumentParser
from dataloader import DataLoader
from model import FastSRGAN
import tensorflow as tf
import os

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
parser.add_argument('--hr_size', default=384, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


@tf.function
def pretrain_step(model, x, y):
    """
    Single step of generator pre-training.
    Args:
        model: A model object with a tf keras compiled generator.
        x: The low resolution image tensor.
        y: The high resolution image tensor.
    """
    with tf.GradientTape() as tape: # GradientTape records operations for automatic differentiation
        fake_hr = model.generator(x)
        loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    grads = tape.gradient(loss_mse, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(grads, model.generator.trainable_variables)) # The zip() function
    # takes iterables, aggregates them in a tuple, and returns it.
    # and gen_optimizer is a class in model, gen_optimizer = keras.optimizers.Adam and apply_gradients is a method
    # of the Adam class that applies gradients to variables in zip(grads, vars)

    return loss_mse


def pretrain_generator(model, dataset, writer):
    """Function that pretrains the generator slightly, to avoid local minima.
    Args:
        model: The keras model to train.
        dataset: A tf dataset object of low and high res images to pretrain over.
        writer: A summary writer object.
    Returns:
        None
    """
    with writer.as_default(): # check where pretrain_generator is called to find out what writer is
        iteration = 0
        for _ in range(1):
            for x, y in dataset:
                loss = pretrain_step(model, x, y)
                if iteration % 20 == 0:
                    tf.summary.scalar('MSE Loss', loss, step=tf.cast(iteration, tf.int64)) # tf.cast changes tensor
                    # to a new data type
                    writer.flush() # flushes the writer buffer i.e. transfers computer data from a temporary
                    # storage area to the computer's permanent memory.
                iteration += 1


@tf.function
def train_step(model, x, y): # this function gives us the various loss at a given step of the network
    """Single train step function for the SRGAN.
    Args:
        model: An object that contains a tf keras compiled discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.

    Returns:
        d_loss: The mean loss of the discriminator.
    """
    # Label smoothing for better gradient flow
    valid = tf.ones((x.shape[0],) + model.disc_patch) # disc_patch = keras.optimizers.schedules.ExponentialDecay
    fake = tf.zeros((x.shape[0],) + model.disc_patch) # fake has shape (x.shape[0],) + model.disc_patch
    # valid = 1 because that's the right one, fake = 0 because it's wrong -- maybe, double check

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From LR image generate HR version
        fake_hr = model.generator(x)

        # Train the discriminators (original images = real; generated = Fake)
        valid_prediction = model.discriminator(y)
        fake_prediction = model.discriminator(fake_hr)

        # Generator loss
        content_loss = model.content_loss(y, fake_hr) # computes the loss using features extracted with VGG19
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction) # cf. written notes
        # fake_prediction is the output of the discriminator applied to the fake HR image
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr) # standard MSE
        perceptual_loss = content_loss + adv_loss + mse_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction) # same reason as adv_loss
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction) # cf. written notes again
        d_loss = tf.add(valid_loss, fake_loss)

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss, mse_loss


#def perceptual_loss(content, adversarial, mse):
#    """Funtion by Tara to be able to compile model"""
#    perceptual = content + adversarial + mse

#    return perceptual


def train(model, dataset, log_iter, writer):
    """
    Function that defines a single training step for the SR-GAN.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in 
                  tensorboard.
        writer: Summary writer
    """
    with writer.as_default():
        # Iterate over dataset
        for x, y in dataset:
            disc_loss, adv_loss, content_loss, mse_loss = train_step(model, x, y)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)
                #model.generator.compile(optimizer=model.gen_optimizer, loss=perceptual_loss(content_loss,adv_loss, mse_loss)) -- added by Tara
                model.generator.save('models/generator.h5')
                model.discriminator.save('models/discriminator.h5')
                writer.flush()
            model.iterations += 1


def main():
    # Parse the CLI arguments.
    args = parser.parse_args()

    # create directory for saving trained models.
    if not os.path.exists('models'):
        os.makedirs('models')

    # Create the tensorflow dataset.
    ds = DataLoader(args.image_dir, args.hr_size).dataset(args.batch_size)

    # Initialize the GAN object.
    gan = FastSRGAN(args)

    # Define the directory for saving pretrainig loss tensorboard summary.
    pretrain_summary_writer = tf.summary.create_file_writer('logs/pretrain')

    # Run pre-training.
    pretrain_generator(gan, ds, pretrain_summary_writer)

    # Define the directory for saving the SRGAN training tensorbaord summary.
    train_summary_writer = tf.summary.create_file_writer('logs/train')

    # Run training.
    for _ in range(args.epochs):
        train(gan, ds, args.save_iter, train_summary_writer)


if __name__ == '__main__':
    main()
