import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common


class Agent(common.Module):
    def __init__(self, config, obs_space, act_space, step):
        super(Agent, self).__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = WorldModel(config, obs_space, self.tfstep)

    @tf.function
    def train(self, data, state=None):
        # only train rssm, vit_encoder, vit_decoder
        metrics = {}
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        return state, metrics

    @tf.function
    def train_mae(self, data):
        # train mae_encoder, mae_decoder
        metrics = {}
        mets = self.wm.train_mae(data)
        metrics.update(mets)
        return metrics

    @tf.function
    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        report["openl_image"] = self.wm.video_pred(data)
        return report


class WorldModel(common.Module):
    def __init__(self, config, obs_space, tfstep):
        super(WorldModel, self).__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.tfstep = tfstep
        # rssm
        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.feat_dim = config.rssm.deter + config.rssm.stoch * config.rssm.discrete
        # mae
        self.mae_encoder, self.mae_decoder = common.mae_factory(**config.mae)
        # ViT for latent dynamics model
        self.wm_vit_encoder, self.wm_vit_decoder = common.flat_vit_factory(**config.wm_flat_vit)
        # Optimizer
        self.model_opt = common.Optimizer("model", **config.model_opt)
        self.mae_opt = common.Optimizer("mae", **config.mae_opt)
        # ImageNet stats
        self.imagenet_mean = tf.constant([0.485, 0.456, 0.406])
        self.imagenet_std = tf.constant([0.229, 0.224, 0.225])

        self.step = 0

    def train(self, data, state=None):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)
        modules = [self.rssm, self.wm_vit_encoder, self.wm_vit_decoder]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def train_mae(self, data):
        with tf.GradientTape() as mae_tape:
            mae_loss, metrics = self.loss_mae(data)
        modules = [self.mae_encoder, self.mae_decoder]
        metrics.update(self.mae_opt(mae_tape, mae_loss, modules))
        return metrics

    def loss(self, data, state=None):
        # train rssm without masking
        data = self.preprocess(data)
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])

        # Forward without masking
        m = 0.0
        latent, mask, _ = self.mae_encoder.forward_encoder(videos, m, T)
        feature = latent
        data["feature"] = tf.stop_gradient(feature.astype(tf.float32))
        # Detach features
        feature = tf.stop_gradient(feature)
        # ViT encoder with average pooling
        ## Move [CLS] to last position
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        wm_latent = self.wm_vit_encoder.forward_encoder(feature)
        embed = wm_latent.mean(1).reshape([B, T, wm_latent.shape[-1]])

        # rssm forward
        # data - mae_encoder -> vit -> rssm
        post, prior = self.rssm.observe(embed, data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        # losses = {}
        # losses["kl"] = tf.clip_by_value(kl_loss * self.config.wmkl.scale, self.config.wmkl_minloss, 100.0).mean()
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)

        # Feature reconstruction loss
        # rssm -> vit_decoder
        feat = tf.reshape(feat, [B * T, 1, self.feat_dim])
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)
        dist = common.MSEDist(tf.cast(feature_pred, tf.float32), 1, "sum")
        like = tf.cast(dist.log_prob(data["feature"]), tf.float32)
        likes["feature"] = like
        losses["feature"] = -like.mean()

        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean()
        metrics["prior_ent"] = self.rssm.get_dist(prior).entropy().mean()
        metrics["post_ent"] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def loss_mae(self, data):
        data = self.preprocess(data)
        key = "image"
        videos = data[key]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])
        losses, metrics = {}, {}

        # MAE forward
        m = self.config.mask_ratio
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, 1)

        if self.config.mae.reward_pred:
            decoder_pred, reward_pred = self.mae_decoder.forward_decoder(
                latent, ids_restore
            )
            # Reward prediction loss
            reward_pred = tf.reshape(reward_pred, [B, T, 1])
            reward = tf.reshape(data["reward"], [B, T, 1])
            reward_loss = self.mae_decoder.forward_reward_loss(reward, reward_pred)
            losses["mae_reward"] = reward_loss
        else:
            decoder_pred = self.mae_decoder.forward_decoder(latent, ids_restore)

        # Image reconstruction loss
        decoder_loss = self.mae_decoder.forward_loss(videos, decoder_pred, mask)
        losses[key] = decoder_loss

        # Summation and log metrics
        mae_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        metrics.update({f"{name}_loss": value for name, value in losses.items()})
        return mae_loss, metrics

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            if value.dtype == tf.uint8:
                value = self.standardize(value.astype(dtype) / 255.0)
            obs[key] = value
        return obs

    @tf.function
    def standardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = (x - mean) / std
        return x

    @tf.function
    def destandardize(self, x):
        mean = tf.cast(self.imagenet_mean, x.dtype)
        std = tf.cast(self.imagenet_std, x.dtype)
        mean = mean.reshape([1] * (len(x.shape) - 1) + [3])
        std = std.reshape([1] * (len(x.shape) - 1) + [3])
        x = x * std + mean
        return x

    @tf.function
    def video_pred(self, data):
        data = {k: v[:6] for k, v in data.items()}
        videos = data["image"]
        B, T, H, W, C = videos.shape
        videos = videos.reshape([B * T, H, W, C])

        # Autoencoder reconstruction
        m = 0.0 if self.config.mae.early_conv else self.config.mask_ratio
        recon_latent, recon_mask, recon_ids_restore = self.mae_encoder.forward_encoder(
            videos, m, T
        )
        recon_model = self.mae_decoder.forward_decoder(recon_latent, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon_model = recon_model[0]  # first element is decoded one
        recon_model = tf.cast(recon_model, tf.float32)
        recon_model = self.mae_decoder.unpatchify(recon_model[: B * T])
        recon_model = tf.cast(
            self.destandardize(recon_model.reshape([B, T, H, W, C])), tf.float32
        )

        # Latent dynamics model prediction
        # 1: Extract MAE representations
        m = 0.0
        latent, mask, ids_restore = self.mae_encoder.forward_encoder(videos, m, T)
        feature = tf.stop_gradient(latent)

        # 2: Reconstructions from conditioning frames
        # 2-1: Process through ViT encoder
        ## Move [CLS] to last position for positional embedding
        feature = tf.concat([feature[:, 1:], feature[:, :1]], axis=1)
        wm_latent = self.wm_vit_encoder.forward_encoder(feature)
        embed = wm_latent.mean(1).reshape([B, T, wm_latent.shape[-1]])

        # 2-2: Process these through RSSM
        # states, _ = self.rssm.observe(embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        states, _ = self.rssm.observe(embed[:6, :5], data["is_first"][:6, :5])
        feat = self.rssm.get_feat(states)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, self.feat_dim])

        # 2-3: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 2-4 Process these through MAE decoder
        recon_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, :5].reshape(
            [b * t, -1]
        )
        recon = self.mae_decoder.forward_decoder(feature_pred, recon_ids_restore)
        if self.config.mae.reward_pred:
            recon = recon[0]
        recon = tf.cast(recon, tf.float32)
        recon = self.mae_decoder.unpatchify(recon[: b * t])
        recon = tf.reshape(
            recon, [b, t, recon.shape[1], recon.shape[2], recon.shape[3]]
        )
        recon = self.destandardize(recon)

        # 3: Open-loop prediction
        # 3-1: Process through RSSM to obtain prior
        init = {k: v[:, -1] for k, v in states.items()}
        # prior = self.rssm.imagine(data["action"][:6, 5:], init)
        prior = self.rssm.imagine(data["is_first"][:6, 5:], init)
        feat = self.rssm.get_feat(prior)
        b, t = feat.shape[0], feat.shape[1]
        feat = tf.reshape(feat, [b * t, 1, self.feat_dim])

        # 3-2: Process through ViT decoder
        feature_pred = self.wm_vit_decoder.forward_decoder(feat)
        ## Move [CLS] to first position
        feature_pred = tf.concat([feature_pred[:, -1:], feature_pred[:, :-1]], axis=1)

        # 3-3: Process these through MAE decoder
        openl_ids_restore = tf.reshape(ids_restore, [B, T, -1])[:6, 5:].reshape(
            [b * t, -1]
        )
        openl = self.mae_decoder.forward_decoder(feature_pred, openl_ids_restore)
        if self.config.mae.reward_pred:
            openl = openl[0]
        openl = tf.cast(openl, tf.float32)
        openl = self.mae_decoder.unpatchify(openl[: b * t])
        openl = tf.reshape(
            openl, [b, t, openl.shape[1], openl.shape[2], openl.shape[3]]
        )
        openl = self.destandardize(openl)

        # Concatenate across timesteps
        model = tf.concat([recon, openl], 1)
        truth = tf.cast(
            self.destandardize(videos.reshape([B, T, H, W, C])[:6]), tf.float32
        )
        video = tf.concat([truth, recon_model, model], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
