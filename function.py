import torch
import torch.nn as nn

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_emd_loss(pred, target):
    b, _, h, w = pred.shape
    pred = pred.reshape([b, -1, w * h])
    pred_norm = torch.sqrt((pred**2).sum(1).reshape([b, -1, 1]))
    pred = pred.transpose(2, 1)
    target_t = target.reshape([b, -1, w * h])
    target_norm = torch.sqrt((target**2).sum(1).reshape([b, 1, -1]))
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity
    return dist

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def mean_normalization(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size))/std.expand(size)
    return normalized_feat
    

class Attention(nn.Module):
    def __init__(self, num_features):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.key_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.value_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.softmax = nn.Softmax(dim = -1)
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.uniform_(self.query_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.uniform_(self.key_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.value_conv.weight)
        nn.init.uniform_(self.value_conv.bias, 0.0, 1.0)
        
    def forward(self, content_feat, style_feat):
        Query = self.query_conv(mean_normalization(content_feat))
        Key = self.key_conv(mean_normalization(style_feat))
        Value = self.value_conv(style_feat)
        batch_size, channels, height_c, width_c = Query.size()
        Query = Query.view(batch_size, -1, width_c * height_c).permute(0, 2, 1)
        batch_size, channels, height_s, width_s = Key.size()
        Key = Key.view(batch_size, -1, width_s * height_s)
        Attention_Weights = self.softmax(torch.bmm(Query, Key))

        Value = Value.view(batch_size, -1, width_s * height_s)
        Output = torch.bmm(Value, Attention_Weights.permute(0, 2, 1))
        Output = Output.view(batch_size, channels, height_c, width_c)
        return Output


class SAFIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.shared_weight = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.shared_bias = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.shared_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.gamma_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.beta_conv = nn.Conv2d(num_features, num_features, (1, 1))
        self.attention = Attention(num_features)
        self.relu = nn.ReLU()
        nn.init.ones_(self.shared_weight)
        nn.init.zeros_(self.shared_bias)
        nn.init.xavier_uniform_(self.gamma_conv.weight)
        nn.init.uniform_(self.gamma_conv.bias, 0.0, 1.0)
        nn.init.xavier_uniform_(self.beta_conv.weight)
        nn.init.uniform_(self.beta_conv.bias, 0.0, 1.0)

    def forward(self, content_feat, style_feat, output_shared=False):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_feat = self.attention(content_feat, style_feat)
        style_gamma = self.relu(self.gamma_conv(style_feat))
        style_beta = self.relu(self.beta_conv(style_feat))
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        shared_affine_feat = normalized_feat * self.shared_weight.view(1, self.num_features, 1, 1).expand(size) + \
                             self.shared_bias.view(1, self.num_features, 1, 1).expand(size)
        if output_shared:
            return shared_affine_feat
        output = shared_affine_feat * style_gamma + style_beta
        return output


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self,
                 gan_mode,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None

        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss_weight = loss_weight

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgan', 'wgangp', 'hinge', 'logistic']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            if not hasattr(self, 'target_real_tensor'):
                self.target_real_tensor = torch.full(
                    prediction.shape,
                    fill_value=self.target_real_label).float().to(device)
            target_tensor = self.target_real_tensor
        else:
            if not hasattr(self, 'target_fake_tensor'):
                self.target_fake_tensor = torch.full(
                    prediction.shape,
                    fill_value=self.target_fake_label).float().to(device)
            target_tensor = self.target_fake_tensor

        return target_tensor

    def __call__(self,
                 prediction,
                 target_is_real,
                 is_disc=False,
                 is_updating_D=None):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode.find('wgan') != -1:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1 - prediction) if is_updating_D else -prediction
            else:
                loss = F.relu(1 + prediction) if is_updating_D else prediction
            loss = loss.mean()
        elif self.gan_mode == 'logistic':
            if target_is_real:
                loss = F.softplus(-prediction).mean()
            else:
                loss = F.softplus(prediction).mean()

        return loss if is_disc else loss * self.loss_weight


class Discriminator(nn.Module):
    def __init__(self, depth=5, num_channels=64):
        super(Discriminator, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3,num_channels,3,stride=1,padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(0.2)
            )
        self.body = []
        for i in range(depth - 2):
            self.body.append(
                nn.Conv2d(num_channels,
                          num_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            self.body.append(nn.BatchNorm2d(num_channels))
            self.body.append(nn.LeakyReLU(0.2))
        self.body = nn.Sequential(*self.body)
        self.tail = nn.Conv2d(num_channels,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.ganloss = GANLoss('lsgan')

    def losses(self, real, fake):
        pred_real = self(real)
        loss_D_real = self.ganloss(pred_real, True)
        pred_fake = self(fake)
        loss_D_fake = self.ganloss(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    @torch.no_grad()
    def gradients(self):
        for p in self.parameters():
            if p.grad is None:
                continue
            yield p

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
