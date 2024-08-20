import random
import matplotlib.pyplot as plt
import nibabel
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from data.dhcploader import DHCPDataset
from torch.utils.data import DataLoader
from nn.network import Network
import numpy as np
import time
import os
import nibabel as nib

def main():
    defaults = OmegaConf.load('defaults.yaml')
    cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(defaults, cli)
    if cli.config:
        config_file = OmegaConf.load(cli.config)
        cfg = OmegaConf.merge(defaults, config_file, cli)
    if not cfg.inference: #cfg.mode == 'train':
        train(cfg)
    else:
        cfg = OmegaConf.merge(config_file, cli)
        evaluate(cfg)




def train(cfg):
    print('training')
    torch.manual_seed(0)
    random.seed(0)
    sw = SummaryWriter(cfg.log_dir)
    cfg.log_dir = sw.log_dir
    training_resumed = cfg.start_epoch > 0 and cfg.log_dir is not None
    if not training_resumed:
        OmegaConf.save(cfg, os.path.join(sw.log_dir, 'config.yaml'))
    else:
        cfg_resume = OmegaConf.load(os.path.join(cfg.log_dir, 'config.yaml'))
        cfg.network = cfg_resume.network
    device = torch.device(cfg.device)
    net = Network(**cfg.network)
    net.to(device)
    opt = torch.optim.AdamW(params=net.parameters(), lr=cfg.lr)
    dataloader, dataset = create_dataloader(cfg)
    validation_loader, dataset_validation = create_dataloader(cfg, mode='validation')
    id_to_latent_idx = dict(common=0)

    global_step = 0
    # load state dict for continuing
    start_epoch = cfg.start_epoch

    if not training_resumed:
        sw.add_text('cfg', '  \n'.join(['\t'+l for l in OmegaConf.to_yaml(cfg).replace('  ', '\t').splitlines()]), global_step=0)
    if training_resumed:
        print(f"resuming from epoch {cfg.start_epoch} from {cfg.log_dir}")
        save_path = os.path.join(sw.log_dir, 'checkpoints')
        net_state_dict_path = os.path.join(save_path, f'state_dict_net_{start_epoch:05d}.pt')
        optim_state_dict_path = os.path.join(save_path, f'state_dict_opt_{start_epoch:05d}.pt')
        latent_dict_path = os.path.join(save_path, f'latent_dict_{start_epoch:05d}.yaml')
        start_epoch += 1
        global_step = start_epoch * len(dataloader) + 1
        try:
            net_state_dict = torch.load(net_state_dict_path, map_location=device)
            net.load_state_dict(net_state_dict)
            opt_state_dict = torch.load(optim_state_dict_path, map_location=device)
            opt.load_state_dict(opt_state_dict)
            id_to_latent_idx = dict(OmegaConf.load(latent_dict_path))
        except FileNotFoundError:
            print('Statedict not Found')
            raise

    loss = loss_fg = loss_bg = None
    for epoch in range(start_epoch, cfg.epochs):
        for iter, data in enumerate(dataloader):
            t = time.time()
            global_step += 1
            if cfg.augment_flip:
                data = augment_flip(data)
            subject_ids = data['subject_id']
            if not cfg.train_ssl:
                data['subject_id'] = data['subject_id'] + data['session_id']
            if cfg.train_sgla:  # for SGLA: overwrite subject_id with the 'common' keyword
                data['subject_id'] = [sid if random.random() >= cfg.train_sgla else 'common' for sid in subject_ids]

            losses, raw = forward_loss(net, data, id_to_latent_idx, cfg.batch_size_pixels_ratio, sampling_foreground_ratio=cfg.sampling_foreground_ratio, device=device)
            opt.zero_grad()
            loss = losses['total']
            loss.backward()
            opt.step()
            # report
            dt = time.time() - t
            sw.add_scalar('loss_total', loss, global_step=global_step)
            loss_bg = losses['background'].item()
            loss_fg = losses['foreground'].item()
            sw.add_scalar('loss_fg', loss_fg, global_step=global_step)
            sw.add_scalar('loss_bg', loss_bg, global_step=global_step)
            sw.add_scalar('time_step', dt, global_step=global_step)
            print(f"{epoch=} {iter=} loss: {loss.item():10.8f} loss_fg: {loss_fg:10.8f} loss_bg: {loss_bg:10.8f}")

            if global_step % cfg.validation_step == 0:
                print('validation')
                with torch.no_grad():
                    # turn network to eval
                    val_img = create_primary_planes(net, data, id_to_latent_idx, device)
                    sw.add_image('validation_planes', val_img, global_step=global_step, dataformats='HW')
                    val_img = create_age_sequence(net, data, id_to_latent_idx, device)
                    sw.add_image('validation_ages', val_img, global_step=global_step, dataformats='HW')
                print('validation done')
                pass

        if epoch % cfg.save_epoch == 0 and epoch > 0:
            print(f'saving epoch {epoch}')
            save_path = os.path.join(sw.log_dir, 'checkpoints')
            os.makedirs(save_path, exist_ok=True)
            net_state_dict_path = os.path.join(save_path, f'state_dict_net_{epoch:05d}.pt')
            optim_state_dict_path = os.path.join(save_path, f'state_dict_opt_{epoch:05d}.pt')
            latent_dict_path = os.path.join(save_path, f'latent_dict_{epoch:05d}.yaml')
            torch.save(net.state_dict(), net_state_dict_path)
            torch.save(opt.state_dict(), optim_state_dict_path)
            OmegaConf.save(config=OmegaConf.create(id_to_latent_idx), f=latent_dict_path)
            #loaded = dict(OmegaConf.load(latent_dict_path))
            #assert loaded == id_to_latent_idx
            print('saved')

        print('epoch done')
    sw.add_hparams(
        hparam_dict={**{k: v for k, v in cfg.items() if k != 'network'}, **{('network.'+k): v for k, v in cfg.network.items()}},
        metric_dict=dict(
            _loss_total=loss,
            _loss_fg=loss_fg,
            _loss_bg=loss_bg,
        ),
        run_name='.',
    )
    print('training done')
    loss_unscaled = 42
    return loss_unscaled

def augment_flip(data):
    if random.random() > 0.5:
        new_data = dict(**data)
        new_data['subject_id'] = [f"{sid}#flip" for sid in new_data['subject_id']]
        new_data['image'] = torch.flip(new_data['image'], dims=(2,)) # [B, C, H, W (D,)]
        new_data['mask'] = torch.flip(new_data['mask'], dims=(1,))   # [B, H, W (D,)]
        return new_data
    return data

##############################################################################################
def evaluate(cfg):
    print('evaluate ' + cfg.mode)
    torch.manual_seed(0)
    random.seed(0)
    assert cfg.start_epoch > 0 and cfg.log_dir is not None
    cfg_resume = OmegaConf.load(os.path.join(cfg.log_dir, 'config.yaml'))
    cfg = OmegaConf.merge(cfg_resume, cfg)
    device = torch.device(cfg.device)
    net = Network(**cfg.network)
    net.to(device)
    opt = torch.optim.AdamW(params=net.parameters(), lr=cfg.lr)
    dataloader, dataset = create_dataloader(cfg)

    # load state dict for continuing
    start_epoch = cfg.start_epoch
    print(f"resuming from epoch {cfg.start_epoch} from {cfg.log_dir}")
    save_path = os.path.join(cfg.log_dir, 'checkpoints')
    net_state_dict_path = os.path.join(save_path, f'state_dict_net_{start_epoch:05d}.pt')
    optim_state_dict_path = os.path.join(save_path, f'state_dict_opt_{start_epoch:05d}.pt')
    latent_dict_path = os.path.join(save_path, f'latent_dict_{start_epoch:05d}.yaml')
    start_epoch += 1
    global_step = start_epoch * len(dataloader) + 1
    try:
        net_state_dict = torch.load(net_state_dict_path, map_location=device)
        net.load_state_dict(net_state_dict)
        opt_state_dict = torch.load(optim_state_dict_path, map_location=device)
        opt.load_state_dict(opt_state_dict)
        id_to_latent_idx = dict(OmegaConf.load(latent_dict_path))
    except FileNotFoundError:
        print('Statedict not Found')
        raise

    losses = []
    output_path = os.path.join(cfg.log_dir, 'results', cfg.mode, str(cfg.start_epoch))
    print(f'saving to {output_path}')
    os.makedirs(output_path, exist_ok=True)
    for iter, data in enumerate(dataloader):
        print('starting inference')

        # find second scan of same subject, and load it from dataset
        df = dataset.database
        scans = df[df['subject_id'].isin(data['subject_id'])]
        if len(scans) != 2:
            print('stopping inference, only one scan found')
            continue  # assuming there are two scans for same subject
        other_session_id = list(set(scans['session_id'])-set(data['session_id']))[0]
        other_data_idx = df.index[(df['session_id'] == other_session_id) & (df['subject_id'].isin(data['subject_id']))].to_list()
        other_data = torch.utils.data.default_collate([dataset[other_data_idx[0]]])

        # perform latent optimization
        t = time.time()
        res = optimize_latent(net, data, device, cfg)
        print(f'optimized latent in {time.time()-t}')
        torch.cuda.empty_cache()
        #res['loss_history']
        #res['latent']
        # find and load second scan of same subject, sample it, compare it

        # also save optimized volume
        supervised_vol = create_volume(net, data, subject_to_latent_idx=None, micro_batch_ratio=cfg.micro_batch_ratio,
                            device=device, mask_image=True, custom_latent=res['latent'])
        supervised_vol = supervised_vol.cpu()
        vol = create_volume(net, other_data, subject_to_latent_idx=None, micro_batch_ratio=cfg.micro_batch_ratio,
                            device=device, mask_image=True, custom_latent=res['latent'])
        vol = vol.cpu()
        print('volume created')
        print(vol.reshape(-1)[0])
        # for debugging: vol = create_volume(net, other_data, subject_to_latent_idx=id_to_latent_idx, device=device, micro_batch_ratio=cfg.micro_batch_ratio,  mask_image=True, custom_latent=None)

        # save all kinds of data and prliminary metrics output
        for b in range(vol.shape[0]):

            # newly generated image with other age
            img_nb = nib.Nifti1Image(vol[b, ...].cpu().numpy(), np.eye(4))
            filename = f"{os.path.basename(other_data['image_path'][b]).removesuffix('.nii.gz')}_from_ses-{data['session_id'][b]}.nii.gz"
            fp = os.path.join(output_path, filename)
            nib.save(img_nb, filename=fp)

            # supervised optimized image with original age
            img_nb = nib.Nifti1Image(supervised_vol[b, ...].cpu().numpy(), np.eye(4))
            filename = f"{os.path.basename(data['image_path'][b]).removesuffix('.nii.gz')}_from_ses-{data['session_id'][b]}.nii.gz"
            fp = os.path.join(output_path, filename)
            nib.save(img_nb, filename=fp)

            orig_image = data['image'][b] * data['mask'][b]
            losses.append(dict(
                subject_id=data['subject_id'][b],
                session_id=other_data['session_id'][b],
                age=other_data['age'][b].item(),
                session_id_from=data['session_id'][b],
                age_from=data['age'][b].item(),
                loss_latent=res['loss_history'][-1]['total'],
                psnr=10*np.log10(1/(orig_image-vol[b, ...]).pow(2).mean().item()),
                psnr_cl=10*np.log10(1/(orig_image-vol[b, ...].clip(0, 1)).pow(2).mean().item()),
            ))

        dt = time.time() - t
        print(f'total time {dt}')
        df = pd.DataFrame(losses)
        df.to_json(os.path.join(output_path, 'data_intermediate.json'))
    df = pd.DataFrame(losses)
    df.to_json(os.path.join(output_path, 'data.json'))
    print('done')



def optimize_latent(net, data, device, cfg):
    latent_size = net.latent_embeddings.embedding_dim
    if False:
        v = net.latent_embeddings.weight.detach().cpu().numpy()
        plt.plot(v[:, 10])
        # svd is interesting
        plt.show()
        print('done')
    latent = torch.zeros((cfg.batch_size, latent_size), requires_grad=True, device=device)
    age_param = torch.zeros((cfg.batch_size,), device=device)+data['age'].item()
    age_param.requires_grad = True
    data['age'] = age_param
    opt = torch.optim.AdamW(params=[
        dict(params=latent, lr=cfg.lr),
    ], lr=cfg.lr)
    net.eval()
    # optimize latent
    loss_history = []
    age_history = []
    for k in range(cfg.iterations_optimize_latent):
        losses, raw = forward_loss(
            net,
            data,
            subject_to_latent_idx=None,
            batch_size_pixels_ratio=cfg.batch_size_pixels_ratio,
            sampling_foreground_ratio=cfg.sampling_foreground_ratio,
            device=device,
            custom_latent=latent,
        )
        latent.grad = None
        losses['total'].backward()
        opt.step()
        if k % 20 == 0:
            print(f'{k=} {losses["foreground"].item()=}')
        loss_history.append({k: v.item() for k, v in losses.items()})
        age_history.append(age_param.item())

    print('latent optimization done')
    net.train()
    out_dict = dict(
        latent=latent.detach(),
        loss_history=loss_history,
    )
    return out_dict

def create_age_sequence(net, data, subject_to_latent_idx, device, custom_latent=None, ages=[0.25, 0.30, 0.35, 0.4, 0.45, 0.5]):
    """creates sequence of ages"""
    with torch.no_grad():
        # turn network to eval
        net.eval()
        out_img = torch.zeros((3*256, 2*256))
        # make a series of images for all the different ages
        for iage, age in enumerate(ages):
            # copy data list
            new_data = {**data}
            new_data['age'] = torch.tensor(age)
            losses, raw = forward_loss(
                net=net,
                data=new_data,
                subject_to_latent_idx=subject_to_latent_idx,
                batch_size_pixels_ratio=None,
                device=device,
                plane=2,
                custom_latent=custom_latent,
            )
            orig = raw['batch_pixel'].cpu().reshape(raw['shape']).squeeze()
            pred = raw['pred_batch_pixel'].cpu().reshape(raw['shape']).squeeze()
            y = iage * (pred.shape[0]//2)
            out_img[y:y+pred.shape[0]//2, 256:256+pred.shape[1]] = pred[:pred.shape[0]//2, ...]
        # first column: place original image according to its age
        y_orig = int((data['age'].clamp(ages[0], ages[-1]) - ages[0])/(ages[-1] - ages[0]) * pred.shape[0]//2 * (len(ages)-1))
        out_img[y_orig:y_orig+(orig.shape[0]//2),   0:    orig.shape[1]] = orig[:orig.shape[0]//2, ...]
        #finally, a difference map between the first and the last age
        y = iage * (pred.shape[0] // 2)
        diff = out_img[y:y+pred.shape[0]//2, 256:256+pred.shape[1]] - out_img[0:0+pred.shape[0]//2, 256:256+pred.shape[1]]
        diff = (diff/2 + 0.5)
        out_img[-pred.shape[0]//2:, 256:256+pred.shape[1]] = diff
        print((diff-0.5).abs().mean())
        net.train()
    return out_img




def create_volume(net, data, subject_to_latent_idx, device='cpu', micro_batch_ratio=1, mask_image=True, custom_latent=None):
    # sample pixels from image, compute forward pass, compute loss
    img = data['image'].to(device)
    mask = data['mask'].to(device)
    if mask_image:
        img = img * mask
    age = data['age'].to(device)

    # create mesh
    device = img.device
    mesh_vs = [age] + [torch.arange(s, device=device) - s/2 for s in img.shape[2:]]
    #mesh_vs = [age] + [(torch.arange(s, device=device) - s/2)*100 for s in img.shape[2:]]  # generate outside content
    coordinate_grid = torch.stack(torch.meshgrid(mesh_vs, indexing='ij'), dim=-1)
    coordinate_grid[..., 1:] /= max(img.shape[2:])  # normalize everything except age, which is already normalized
    coordinate_grid_flat = coordinate_grid.reshape(coordinate_grid.shape[0], -1, coordinate_grid.shape[-1])
    n = img[0, ...].numel() #np.prod(img.shape[1:])

    # get latent vector if needed
    kwargs = dict()
    if custom_latent is None:  # default sampling during training
        for sid in data['subject_id']:
            if sid not in subject_to_latent_idx:
                subject_to_latent_idx[sid] = len(subject_to_latent_idx)
        latent_idx = torch.tensor([subject_to_latent_idx[sid] for sid in data['subject_id']], dtype=int, device=device)
        kwargs.update(latent_idx=latent_idx, custom_latent=None)
    else:  # for inference, when we optimize a custom latent vector
        kwargs.update(latent_idx=None, custom_latent=custom_latent)

    # decompose whole volume into microbatches, as we cannot run the inference
    # for the whole volume in one go
    micro_batch_dim = coordinate_grid_flat.shape[1]
    micro_batch_size = int(micro_batch_dim * micro_batch_ratio)
    outputs = []
    compute_grad = False
    with torch.set_grad_enabled(compute_grad):
        for mb_slice in (slice(i, i+micro_batch_size) for i in range(0, micro_batch_dim, micro_batch_size)):
            out = net(coordinate_grid_flat[:, mb_slice, :], **kwargs)
            if compute_grad:
                out.sum().backward()  # this was just for demonstration purposes, it makes no sense
            outputs.append(out)
    output = torch.cat(outputs, dim=1).reshape(coordinate_grid.shape[:-1])
    #plt.imshow(torch.einsum('ij->ji', output.cpu()[0, ...].max(dim=0)[0]), cmap='inferno', origin='lower')
    #plt.show()
    #plt.imshow(torch.einsum('ij->ji', output.cpu()[0, ...].sum(dim=0)), cmap='inferno', origin='lower')
    #plt.show()
    return output


def create_primary_planes(net, data, subject_to_latent_idx, device, custom_latent=None):
    with torch.no_grad():
        # turn network to eval
        net.eval()
        out_img = torch.zeros((3*256, 2*256))
        for plane in range(3):
            losses, raw = forward_loss(
                net=net,
                data=data,
                subject_to_latent_idx=subject_to_latent_idx,
                batch_size_pixels_ratio=None,
                sampling_foreground_ratio=None,
                device=device,
                plane=plane,
                custom_latent=custom_latent,
            )
            orig = raw['batch_pixel'].cpu().reshape(raw['shape']).squeeze(dim=(0, 1))
            pred = raw['pred_batch_pixel'].cpu().reshape(raw['shape']).squeeze(dim=(0, 1))
            y = plane*256
            out_img[y:y+orig.shape[0],   0:    orig.shape[1]] = orig
            out_img[y:y+pred.shape[0], 256:256+pred.shape[1]] = pred
        net.train()
    return out_img

def create_dataloader(cfg, mode='train'):
    dataset = DHCPDataset(
        directory=cfg.data_dir,
        database=dict(
            train=cfg.database_train,
            validation=cfg.database_validation,
            inference=cfg.database_inference
        )[mode],
        normalize=(lambda x: x/x.max()),
        train_ssl=cfg.train_ssl,
        cache_images=cfg.cache_images,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=mode == 'train',
        num_workers=0,
    )
    return loader, dataset

def create_network(cfg):
    from nn.models import ImplicitNetSegPrior
    net = ImplicitNetSegPrior(cfg)
    return net


def forward_loss(
        net, data, subject_to_latent_idx, batch_size_pixels_ratio, sampling_foreground_ratio=None,
        device='cpu', plane=None, mask_image=True, custom_latent=None):
    # sample pixels from image, compute forward pass, compute loss
    img = data['image'].to(device)
    mask = data['mask'].to(device)
    if mask_image:
        img = img * mask
    age = data['age'].to(device)
    raw = dict()

    batch_pixel, batch_coordinates, sample_raw = sample_pixels(
        img=img,
        age=age,
        foreground_mask=mask,
        batch_size_pixels_ratio=batch_size_pixels_ratio,
        sampling_foreground_ratio=sampling_foreground_ratio,
        plane=plane,
    )
    # get subject index for network
    if custom_latent is None:  # default sampling during training
        for sid in data['subject_id']:
            if sid not in subject_to_latent_idx:
                subject_to_latent_idx[sid] = len(subject_to_latent_idx)
        latent_idx = torch.tensor([subject_to_latent_idx[sid] for sid in data['subject_id']], dtype=int, device=device)
        pred_batch_pixel = net(batch_coordinates, latent_idx=latent_idx, custom_latent=None)
    else:  # for inference, when we optimize a custom latent vector
        pred_batch_pixel = net(batch_coordinates, latent_idx=None, custom_latent=custom_latent)

    loss = (batch_pixel - pred_batch_pixel).pow(2)
    loss_foreground = loss[sample_raw['batch_pixel_foreground']].mean()
    loss_background = loss[~sample_raw['batch_pixel_foreground']].mean()
    loss = loss.mean()
    losses = dict(
        total=loss,
        foreground=loss_foreground,
        background=loss_background,
    )
    raw.update(
        pred_batch_pixel=pred_batch_pixel,
        batch_pixel=batch_pixel,
        batch_coordinates=batch_coordinates,
        masks=sample_raw['sampling_mask'],
        shape=sample_raw['shape'],
    )
    return losses, raw


def sample_pixels(img, age, foreground_mask=None, batch_size_pixels_ratio=1.0, sampling_foreground_ratio=None, plane=None):
    """
    do not process whole image, but just a percentage of the pixels
    batch_size_pixels_ratio defines how many pixels are sampled per image for the training
    sampling_foreground_ratio defines how many of the sampled pixels should be in the foreground (nonzero)
    """
    raw = dict()
    device = img.device
    mesh_vs = [age] + [torch.arange(s, device=device) - s/2 for s in img.shape[2:]]
    coordinate_grid = torch.stack(torch.meshgrid(mesh_vs, indexing='ij'), dim=-1)
    coordinate_grid[..., 1:] /= max(img.shape[2:])  # normalize everything except age, which is already normalized
    # coordinate_grid.shape = img.shape[2:]
    n = img[0, ...].numel()
    if plane is None:  # default, randomly samples
        if sampling_foreground_ratio is None:
            sampling_mask = torch.stack([
                torch.randperm(n).reshape(img.shape[1:])/n <= batch_size_pixels_ratio for _ in range(img.shape[0])
            ], dim=0)
        else:  # set exact amount for foreground and background
            n_sample = int(n * batch_size_pixels_ratio)
            if foreground_mask is None:
                foreground_mask = img != 0
            n_foreground = torch.einsum('i...->i',foreground_mask)  # number of foreground pixels
            n_sample_foreground = torch.minimum(torch.tensor(n_sample * sampling_foreground_ratio, dtype=int), n_foreground+10000000000)
            n_sample_background = n_sample - n_sample_foreground
            sampling_mask = torch.zeros_like(foreground_mask, dtype=torch.bool)
            for b in range(img.shape[0]):
                vf = torch.randperm(    n_foreground[b, ...], device=device) < n_sample_foreground[b]
                sampling_mask[b,  foreground_mask[b, ...]] = vf
                vb = torch.randperm(n - n_foreground[b, ...], device=device) < n_sample_background[b]
                sampling_mask[b, ~foreground_mask[b, ...]] = vb
            sampling_mask = sampling_mask.unsqueeze(1)

        shape = None
    else: # just sample one image plane
        # plane+1 because first dimension is time
        sampling_mask = coordinate_grid[..., plane+1] == coordinate_grid[..., plane+1].median()
        sampling_mask = sampling_mask.unsqueeze(1)
        # use shape for reconstruction
        shape = list(sampling_mask.shape)
        shape.pop(plane+2)
    batch_pixel = img[sampling_mask].reshape(img.shape[0], -1, img.shape[1])
    batch_pixel_foreground = foreground_mask[:, None, ...][sampling_mask].reshape(img.shape[0], -1, img.shape[1])
    batch_coordinates = coordinate_grid[sampling_mask[:, 0, ...], :].reshape(img.shape[0], -1, coordinate_grid.shape[-1])

    raw['batch_pixel_foreground'] = batch_pixel_foreground
    raw['sampling_mask'] = sampling_mask
    raw['shape'] = shape

    return batch_pixel, batch_coordinates, raw





if __name__ == '__main__':
    main()
