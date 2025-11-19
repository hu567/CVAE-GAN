import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # 使用标准负号
import os
from datetime import datetime
import scipy.io  # 用于保存 .mat 文件

# 工具函数
def plot_losses(loss_history, loss_dir):
    """绘制所有损失曲线"""
    epochs_range = range(1, len(loss_history['g_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # 生成器总损失
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, loss_history['g_loss'], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Total Loss')
    plt.legend()
    plt.grid(True)
    
    # 判别器损失
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, loss_history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)
    
    # MSE损失
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, loss_history['mse_loss'], label='MSE Loss')
    plt.plot(epochs_range, loss_history['test_mse'], label='Test MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Reconstruction MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # KL散度损失
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, loss_history['kld_loss'], label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    plt.grid(True)
    
    # GAN损失
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range, loss_history['gan_loss'], label='GAN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Loss')
    plt.legend()
    plt.grid(True)
    
    # 梯度惩罚
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, loss_history['gp'], label='Gradient Penalty')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty')
    plt.title('Gradient Penalty')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, 'loss_curves.png'))
    plt.close()


def save_model(model, model_dir, model_name):
    """保存模型"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, model_name))

def plot_comparison_grid(real_acc, fake_acc, epoch, fig_dir, dataset_type="train"):
    """绘制真实波形与生成波形的对比图，区分训练集和测试集"""
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i >= len(real_acc):
            break
        ax.plot(real_acc[i, :, 0], label="真实波形")
        ax.plot(fake_acc[i, :, 0], label="生成波形")
        ax.legend()
        ax.set_title(f"样本 {i+1}")
        ax.set_xlabel("时间")
        ax.set_ylabel("幅度")
    
    plt.suptitle(f"第 {epoch+1} 轮: 真实波形 vs 生成波形 ({dataset_type})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"epoch_{epoch+1}_comparison_{dataset_type}.png"))
    plt.close()

# 数据加载函数
def load_data(file_path, dataset_type="train"):
    """从HDF5文件加载训练或测试数据，不含fname，并在加载时归一化，返回 tf.data.Dataset"""
    with h5py.File(file_path, 'r') as f:
        spe_group = f['acceleration_spectra']
        wave_group = f['data']
        catalog_group = f['catalog/table']
        depth_km = catalog_group['depth_km'][()]
        distance_km = catalog_group['distance_km'][()]
        network = catalog_group['network'][()]
        
        if isinstance(depth_km[0], bytes):
            depth_km = np.array([float(x.decode('utf-8')) for x in depth_km], dtype=np.float32)
        else:
            depth_km = np.array(depth_km, dtype=np.float32)
        
        if isinstance(distance_km[0], bytes):
            distance_km = np.array([float(x.decode('utf-8')) for x in distance_km], dtype=np.float32)
        else:
            distance_km = np.array(distance_km, dtype=np.float32)
        
        waves = []
        spes = []
        total_samples = len(wave_group)
        print(f"加载 {dataset_type} 数据，总共有 {total_samples} 个样本")
        
        for dataset_name in tqdm(wave_group.keys(), desc=f"📥 加载 {dataset_type} 数据", unit="样本", colour='green'):
            waveform = wave_group[dataset_name][()].astype(np.float32)
            single_wave = waveform[:, 0:1]
            max_abs_wave = np.max(np.abs(single_wave))
            single_wave = single_wave / max_abs_wave
            waves.append(single_wave)
            
            spe = spe_group[dataset_name][()].astype(np.float32)
            single_spe = spe[:, 0]
            max_abs_spe = np.max(np.abs(single_spe))
            single_spe = single_spe / max_abs_spe
            spes.append(single_spe)
        
        waves = np.stack(waves, axis=0)
        spes = np.stack(spes, axis=0)
        
        assert len(depth_km) == total_samples, f"depth_km 长度不匹配: {len(depth_km)} vs {total_samples}"
        
        print(f"{dataset_type} Waves shape: {waves.shape}, Range: [{waves.min():.3f}, {waves.max():.3f}]")
        print(f"{dataset_type} Spes shape: {spes.shape}, Range: [{spes.min():.3f}, {spes.max():.3f}]")
        print(f"{dataset_type} Depth_km shape: {depth_km.shape}, Range: [{depth_km.min():.3f}, {depth_km.max():.3f}]")
        print(f"{dataset_type} Distance_km shape: {distance_km.shape}, Range: [{distance_km.min():.3f}, {distance_km.max():.3f}]")
        print(f"{dataset_type} Network shape: {network.shape}")
        
        if isinstance(network[0], bytes):
            network = np.array([x.decode('utf-8') for x in network])
        
        dataset = tf.data.Dataset.from_tensor_slices((waves, spes, depth_km, distance_km, network))
        if dataset_type == "train":
            dataset = dataset.shuffle(buffer_size=total_samples).batch(128)
        else:
            dataset = dataset.batch(128)
        return dataset

# 生成器模型
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.enc_conv1 = layers.Conv1D(64, 10, strides=2, padding='same')
        self.enc_bn1 = layers.BatchNormalization()
        self.enc_conv2 = layers.Conv1D(128, 10, strides=2, padding='same')
        self.enc_bn2 = layers.BatchNormalization()
        self.enc_conv3 = layers.Conv1D(256, 10, strides=2, padding='same')
        self.enc_bn3 = layers.BatchNormalization()
        self.enc_gap = layers.GlobalAveragePooling1D()
        self.enc_fc_mu = layers.Dense(100)
        self.enc_fc_logvar = layers.Dense(100)
        
        self.dec_fc = layers.Dense(375 * 256)
        self.dec_reshape = layers.Reshape((375, 256))
        self.dec_upsample1 = layers.UpSampling1D(size=2)
        self.dec_conv1 = layers.Conv1D(128, 3, padding='same')
        self.dec_bn1 = layers.BatchNormalization()
        self.dec_upsample2 = layers.UpSampling1D(size=2)
        self.dec_conv2 = layers.Conv1D(64, 3, padding='same')
        self.dec_bn2 = layers.BatchNormalization()
        self.dec_upsample3 = layers.UpSampling1D(size=2)
        self.dec_conv3 = layers.Conv1D(32, 3, padding='same')
        self.dec_bn3 = layers.BatchNormalization()
        self.dec_conv4 = layers.Conv1D(1, 3, padding='same')

    def encode(self, wave, spe):
        x = self.enc_conv1(wave)
        x = self.enc_bn1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.enc_gap(x)
        x = layers.Concatenate()([x, spe])
        mu = self.enc_fc_mu(x)
        log_var = self.enc_fc_logvar(x)
        return mu, log_var

    def decode(self, z, spe):
        x = layers.Concatenate()([z, spe])
        x = self.dec_fc(x)
        x = self.dec_reshape(x)
        x = self.dec_upsample1(x)
        x = self.dec_conv1(x)
        x = self.dec_bn1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dec_upsample2(x)
        x = self.dec_conv2(x)
        x = self.dec_bn2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dec_upsample3(x)
        x = self.dec_conv3(x)
        x = self.dec_bn3(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dec_conv4(x)
        x = tf.keras.layers.Activation('tanh')(x)
        return x

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def call(self, inputs, training=False):
        wave, spe = inputs
        mu, log_var = self.encode(wave, spe)
        z = self.reparameterize(mu, log_var)
        recon_wave = self.decode(z, spe)
        return recon_wave, mu, log_var

    def generate_waveform(self, spe):
        z = tf.random.normal([spe.shape[0], 100])
        return self.decode(z, spe)

# 判别器模型
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_psa = layers.Dense(3000)
        self.reshape_psa = layers.Reshape((3000, 1))
        self.conv1 = layers.Conv1D(32, 5, strides=2, padding='same')
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)
        self.conv2 = layers.Conv1D(64, 5, strides=2, padding='same')
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)
        self.conv3 = layers.Conv1D(128, 5, strides=2, padding='same')
        self.leaky_relu3 = layers.LeakyReLU(alpha=0.2)
        self.conv4 = layers.Conv1D(256, 5, strides=2, padding='same')
        self.leaky_relu4 = layers.LeakyReLU(alpha=0.2)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=False):
        acc, PSA = inputs
        psa = self.fc_psa(PSA)
        psa = self.reshape_psa(psa)
        x = layers.Concatenate(axis=2)([acc, psa])
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)
        x = self.conv4(x)
        x = self.leaky_relu4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 损失函数
def generator_loss(recon_x, x, mu, log_var, discriminator_output, lambda_recon, lambda_kl, lambda_adv):
    mse_loss = tf.reduce_mean(tf.square(recon_x - x))
    kld_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    gan_loss = -tf.reduce_mean(discriminator_output)
    total_loss = lambda_recon * mse_loss + lambda_kl * kld_loss + lambda_adv * gan_loss
    return total_loss, mse_loss, kld_loss, gan_loss

def discriminator_loss(real_output, fake_output, gradient_penalty, lambda_gp):
    d_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gradient_penalty
    return d_loss

def compute_gradient_penalty(D, real_samples, fake_samples, PSA):
    alpha = tf.random.uniform([real_samples.shape[0], 1, 1], 0., 1.)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        d_interpolates = D([interpolates, PSA])
    gradients = tape.gradient(d_interpolates, interpolates)
    gradients = tf.reshape(gradients, [gradients.shape[0], -1])
    gradient_penalty = tf.reduce_mean((tf.norm(gradients, axis=1) - 1) ** 2)
    return gradient_penalty

# 超参数
device = 'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'
n_critic = 2
lambda_recon = 5
lambda_kl = 0.5
lambda_adv = 0.1
lambda_gp = 10
latent_dim = 100
epochs = 200

# 数据路径
train_file = 'H:\\gm-data\\train_data_no_fname_raw.h5'
test_file = 'H:\\gm-data\\test_data_no_fname_raw.h5'

# 加载数据
train_dataset = load_data(train_file, dataset_type="train")
test_dataset = load_data(test_file, dataset_type="test")

# 初始化模型
generator = Generator()
discriminator = Discriminator()

wave_sample = tf.convert_to_tensor(np.random.randn(1, 3000, 1), dtype=tf.float32)
spe_sample = tf.convert_to_tensor(np.random.randn(1, 200), dtype=tf.float32)
inputs = [wave_sample, spe_sample]
generator.build(input_shape=[(None, 3000, 1), (None, 200)])
out_g = generator(inputs)
print('生成器输出形状:', out_g[0].shape)
generator.summary()

out_d = discriminator(inputs)
discriminator.build(input_shape=[(None, 3000, 1), (None, 200)])
print('判别器输出形状:', out_d.shape)
discriminator.summary()

# 优化器
optimizer_G = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
optimizer_D = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)

# 用于可视化的固定样本（训练集和测试集）
fixed_train_samples = next(iter(train_dataset))
fixed_train_real_samples = fixed_train_samples[0][:9].numpy()  # waves from train
fixed_train_PSA_samples = fixed_train_samples[1][:9]  # spes from train

fixed_test_samples = next(iter(test_dataset))
fixed_test_real_samples = fixed_test_samples[0][:9].numpy()  # waves from test
fixed_test_PSA_samples = fixed_test_samples[1][:9]  # spes from test

# 训练循环
model_dir = 'cvaegan_results/save_models_no_fname'
fig_dir = 'cvaegan_results/figures_no_fname'
loss_dir = 'cvaegan_results/losses_no_fname'
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)

# 用于存储所有轮次的损失
loss_history = {
    'g_loss': [],
    'd_loss': [],
    'mse_loss': [],
    'kld_loss': [],
    'gan_loss': [],
    'gp': [],
    'test_mse': []
}

for epoch in range(epochs):
    g_loss_total, d_loss_total = 0, 0
    mse_total, kld_total, gan_total, gp_total = 0, 0, 0, 0
    d_updates = 0
    
    for real_acc, PSA, _, _, _ in train_dataset:
        real_acc = tf.convert_to_tensor(real_acc)
        PSA = tf.convert_to_tensor(PSA)
        batch_size = real_acc.shape[0]
        
        for _ in range(n_critic):
            with tf.GradientTape() as tape_D:
                fake_acc = generator.generate_waveform(PSA)
                d_real = discriminator([real_acc, PSA])
                d_fake = discriminator([fake_acc, PSA])
                gradient_penalty = compute_gradient_penalty(discriminator, real_acc, fake_acc, PSA)
                d_loss = discriminator_loss(d_real, d_fake, gradient_penalty, lambda_gp)
                if d_updates % 200 == 0:
                    print(f"第 {epoch+1} 轮, 批次 {d_updates}: d_real: {tf.reduce_mean(d_real):.3f}, d_fake: {tf.reduce_mean(d_fake):.3f}")
            grads_D = tape_D.gradient(d_loss, discriminator.trainable_variables)
            optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))
            d_loss_total += d_loss.numpy()
            gp_total += gradient_penalty.numpy()
            d_updates += 1
        
        with tf.GradientTape() as tape_G:
            recon_acc, mu, log_var = generator([real_acc, PSA])
            d_fake = discriminator([recon_acc, PSA])
            g_loss, mse_loss, kld_loss, gan_loss = generator_loss(
                recon_acc, real_acc, mu, log_var, d_fake, lambda_recon, lambda_kl, lambda_adv
            )
        grads_G = tape_G.gradient(g_loss, generator.trainable_variables)
        optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        g_loss_total += g_loss.numpy()
        mse_total += mse_loss.numpy()
        kld_total += kld_loss.numpy()
        gan_total += gan_loss.numpy()
    
    g_loss_avg = g_loss_total / len(train_dataset)
    d_loss_avg = d_loss_total / d_updates
    mse_avg = mse_total / len(train_dataset)
    kld_avg = kld_total / len(train_dataset)
    gan_avg = gan_total / len(train_dataset)
    gp_avg = gp_total / d_updates
    
    print(f'第 [{epoch+1}/{epochs}] 轮')
    print(f'  训练集 - 生成器损失: {g_loss_avg:.3f} (MSE: {mse_avg:.3f}, KLD: {kld_avg:.3f}, GAN: {gan_avg:.3f})')
    print(f'  训练集 - 判别器损失: {d_loss_avg:.3f} (梯度惩罚: {gp_avg:.3f})')
    
    test_mse_total = 0
    for real_acc, PSA, _, _, _ in test_dataset:
        recon_acc, _, _ = generator([real_acc, PSA])
        test_mse = tf.reduce_mean(tf.square(recon_acc - real_acc))
        test_mse_total += test_mse.numpy()
    
    test_mse_avg = test_mse_total / len(test_dataset)
    print(f'  测试集 - 重建MSE: {test_mse_avg:.3f}')
    
    # 保存损失到历史记录
    loss_history['g_loss'].append(g_loss_avg)
    loss_history['d_loss'].append(d_loss_avg)
    loss_history['mse_loss'].append(mse_avg)
    loss_history['kld_loss'].append(kld_avg)
    loss_history['gan_loss'].append(gan_avg)
    loss_history['gp'].append(gp_avg)
    loss_history['test_mse'].append(test_mse_avg)
    
    # 每轮保存模型和可视化
    save_model(generator, model_dir, f"generator_epoch_{epoch+1}")
    save_model(discriminator, model_dir, f"discriminator_epoch_{epoch+1}")
    
    fixed_train_fake_samples = generator.generate_waveform(fixed_train_PSA_samples).numpy()
    plot_comparison_grid(fixed_train_real_samples, fixed_train_fake_samples, epoch, fig_dir, dataset_type="train")
    
    fixed_test_fake_samples = generator.generate_waveform(fixed_test_PSA_samples).numpy()
    plot_comparison_grid(fixed_test_real_samples, fixed_test_fake_samples, epoch, fig_dir, dataset_type="test")

# 保存所有轮次的损失到单个 .mat 文件
scipy.io.savemat(os.path.join(loss_dir, 'losses_all_epochs.mat'), loss_history)
print(f"所有损失已保存到 {os.path.join(loss_dir, 'losses_all_epochs.mat')}")
# 绘制并保存损失曲线
plot_losses(loss_history, loss_dir)
print(f"损失曲线图已保存到 {os.path.join(loss_dir, 'loss_curves.png')}")
