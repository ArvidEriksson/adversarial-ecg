from math import ceil 
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import scipy.signal as sgn
import autopgd.autopgd_base as autopgd_base

def pgd_attack(model, traces, labels, device, loss_function=nn.BCEWithLogitsLoss(), eps=4e-3, alpha=1e-3, steps=10):
    traces = traces.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    loss = loss_function
    adv_tracings = traces.clone().detach()
    
    # implement random start?
    
    for _ in range(steps):
        adv_tracings.requires_grad = True
        outputs = model(adv_tracings)
        
        cost = loss(outputs, labels)
        
        grad = torch.autograd.grad(cost, adv_tracings, retain_graph=False, create_graph=False)[0]
        
        adv_tracings = adv_tracings.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_tracings - traces, min=-eps, max=eps)
        adv_tracings = (traces + delta).detach()
        
    return adv_tracings

class PGD():
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    
    
lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_ecgs(
        ecg_1,
        ecg_2, 
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        show_zoom      = False,
        zoom_rate      = 2,
        zoom_box       = [1, 3, -1, 1] # x1, x2, y1, y2
        ):
    """Plot two multi lead ECG charts overlapping.
    # Arguments
        ecg1        : m x n ECG signal data, which m is number of leads and n is length of signal.
        ecg2        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
        show_zoom      : show zoom
        zoom_rate      : zoom rate
        zoom_box       : placement of zoom box
    """

    if not lead_order:
        lead_order = list(range(0,len(ecg_1)))
    secs  = len(ecg_1[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns))
    # display_factor = 2.5
    display_factor = 1
    line_width = 0.5
    fig_width = secs * columns * display_factor
    fig_height = rows * row_height / 5 * display_factor
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0,
        left   = 0,  # the left side of the subplots of the figure
        right  = 1,  # the right side of the subplots of the figure
        bottom = 0,  # the bottom of the subplots of the figure
        top    = 1
        )

    fig.suptitle(title)

    x_min = 0
    x_max = columns*secs
    y_min = row_height/4 - (rows/2)*row_height
    y_max = row_height/4

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line_1  = (0,0,1)
        color_line_2 = (1,0,0)
    else:
        color_major = (1,0,0)
        color_minor = (1, 0.7, 0.7)
        color_line_1  = (0,0,0.7)
        color_line_2 = (0,0.7,0)

    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))

        ax.minorticks_on()
        
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)


    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c
                    if(show_separate_line):
                        ax.plot([x_offset, x_offset], [ecg_2[t_lead][0] + y_offset - 0.3, ecg_2[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line_2)
                        ax.plot([x_offset, x_offset], [ecg_1[t_lead][0] + y_offset - 0.3, ecg_1[t_lead][0] + y_offset + 0.3], linewidth=line_width * display_factor, color=color_line_1)

                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)
                ax.plot(
                    np.arange(0, len(ecg_2[t_lead])*step, step) + x_offset, 
                    ecg_2[t_lead] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line_2
                    )
                ax.plot(
                    np.arange(0, len(ecg_1[t_lead])*step, step) + x_offset, 
                    ecg_1[t_lead] + y_offset,
                    linewidth=line_width * display_factor, 
                    color=color_line_1
                    )
                if(show_zoom):
                    # zoom in on the middle of the graph
                    # we want the inset to have the same ratio as the main plot
                    y_width = y_max - y_min
                    x_width = x_max - x_min
                    
                    print(y_width, x_width)
                    # show the ratio of the main plot
                    # density along x axis is x_width / fig_width
                    # density along y axis is y_width / fig_height
                    # therefore the inset should be scaled by fig_width / x_width and fig_height / y_width
                    
                    #axins = ax.inset_axes([x_offset + 0.5, y_offset + 1.5, (zoom_box[1] - zoom_box[0])/fig_width, (zoom_box[3] - zoom_box[2])/fig_height*y_width/x_width])
                    axins = ax.inset_axes([x_offset + 0.5, y_offset + 1.5, (zoom_box[1] - zoom_box[0])/x_width*zoom_rate, (zoom_box[3] - zoom_box[2])/y_width*zoom_rate])
                    axins.plot(
                        np.arange(0, len(ecg_2[t_lead])*step, step) + x_offset, 
                        ecg_2[t_lead] + y_offset,
                        linewidth=line_width * display_factor, 
                        color=color_line_2
                        )
                    axins.plot(
                        np.arange(0, len(ecg_1[t_lead])*step, step) + x_offset, 
                        ecg_1[t_lead] + y_offset,
                        linewidth=line_width * display_factor, 
                        color=color_line_1
                        )
                    axins.set_xlim(zoom_box[0], zoom_box[1])
                    axins.set_ylim(zoom_box[2], zoom_box[3])
                    axins.set_xticks([])
                    axins.set_yticks([])
                    ax.indicate_inset_zoom(axins)
                    
def train_loop(epoch, dataloader, model, optimizer, loss_function, device, adversarial=False, adv_eps=5e-2, adv_alpha=1e-2, adv_steps=10):
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # progress bar def
    train_pbar = tqdm(dataloader, desc=f"Training epoch {epoch:2d}", leave=True)
    # training loop
    for traces, diagnoses in train_pbar:
        optimizer.zero_grad()
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces.to(device), diagnoses.to(device)

        # forward pass
        output = model(traces)
        
        # adversarial training
        if adversarial:
            adv_traces = pgd_attack(model, traces, diagnoses, device, loss_function=loss_function, eps=adv_eps, alpha=adv_alpha, steps=adv_steps)
            adv_output = model(adv_traces)
            
            # let loss be the average of the original and adversarial loss
            loss = (loss_function(output, diagnoses) + loss_function(adv_output, diagnoses)) / 2
            
        else:
            loss = loss_function(output, diagnoses)
                
        # backward pass
        loss.backward()
        optimizer.step()

        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()
    return total_loss / n_entries

def train_loop_apgd(epoch, dataloader, model, optimizer, loss_function, device, adversarial=False, adv_eps=5e-2, adv_iters=10, adv_restarts=1):
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    attack = autopgd_base.APGDAttack(model, n_iter=adv_iters, norm='Linf', n_restarts=adv_restarts, eps=adv_eps, seed=0, loss='bce', eot_iter=1, rho=.75, device=device)
    # progress bar def
    train_pbar = tqdm(dataloader, desc=f"Training epoch {epoch:2d}", leave=True)
    # training loop
    for traces, diagnoses in train_pbar:
        optimizer.zero_grad()
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces.to(device), diagnoses.to(device)

        # forward pass
        output = model(traces)
        
        # adversarial training
        if adversarial:
            model.eval()
            adv_traces = attack.perturb(traces,diagnoses,best_loss=True)
            model.train()
            adv_output = model(adv_traces)
            
            # let loss be the average of the original and adversarial loss
            loss = (loss_function(output, diagnoses) + loss_function(adv_output, diagnoses)) / 2
            
        else:
            loss = loss_function(output, diagnoses)
                
        # backward pass
        loss.backward()
        optimizer.step()

        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()
    return total_loss / n_entries

def eval_loop_apgd(epoch, dataloader, model, loss_function, device, adversarial=False, adv_eps=5e-2, adv_iters=10, adv_restarts=1, post_process=None, post_process_args=None):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    attack = autopgd_base.APGDAttack(model, n_iter=adv_iters, norm='Linf', n_restarts=adv_restarts, eps=adv_eps, seed=0, loss='bce', eot_iter=1, rho=.75, device=device)
    valid_pred, valid_true = [], []
    # progress bar def
    eval_pbar = tqdm(dataloader, desc=f"Validating epoch {epoch:2d}", leave=True)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        old_traces = traces.clone().detach()
        
        if adversarial:
            # generate adversarial samples
            traces = attack.perturb(traces,diagnoses,best_loss=True)
            # check if old traces are the same as the new ones
            if torch.allclose(old_traces.cpu(), traces.cpu(), atol=1e-7):
                print("WARNING: Adversarial attack failed")

            # if using some post processing, i.e. smoothing (WIP)
            if post_process is not None:
                #traces = filter_adversarial(traces, 400, fc=100)
                traces = traces.cpu().numpy()
                traces = [post_process(np.transpose(t), *post_process_args) for t in traces]
                traces = [np.transpose(t) for t in traces]
                # make traces into ndarray
                traces = np.array(traces)
                traces = torch.tensor(traces, dtype=torch.float32).to(device)

        with torch.no_grad():
            # forward pass
            output = model(traces)
            loss = loss_function(output, diagnoses)
            
            # save predictions
            valid_pred.append(output.detach().cpu().numpy())
            valid_true.append(diagnoses.detach().cpu().numpy())

            # Update accumulated values
            total_loss += loss.detach().cpu().numpy()
            n_entries += len(traces)

        # Update progress bar
        eval_pbar.set_postfix({'loss': total_loss / n_entries})
    eval_pbar.close()
    return total_loss / n_entries, np.vstack(valid_pred), np.vstack(valid_true)

def eval_loop(epoch, dataloader, model, loss_function, device, adversarial=False, adv_eps=4e-2, adv_alpha=1e-2, adv_steps=10, post_process=None, post_process_args=None):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    valid_pred, valid_true = [], []
    # progress bar def
    eval_pbar = tqdm(dataloader, desc=f"Validating epoch {epoch:2d}", leave=True)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        
        if adversarial:
            # generate adversarial samples
            traces = pgd_attack(model, traces, diagnoses, device, eps=adv_eps, alpha=adv_alpha, steps=adv_steps)
            
            # if using some post processing, i.e. smoothing (WIP)
            if post_process is not None:
                #traces = filter_adversarial(traces, 400, fc=100)
                traces = traces.cpu().numpy()
                traces = [post_process(np.transpose(t), *post_process_args) for t in traces]
                traces = [np.transpose(t) for t in traces]
                # make traces into ndarray
                traces = np.array(traces)
                traces = torch.tensor(traces, dtype=torch.float32).to(device)

        with torch.no_grad():
            # forward pass
            output = model(traces)
            loss = loss_function(output, diagnoses)
            
            # save predictions
            valid_pred.append(output.detach().cpu().numpy())
            valid_true.append(diagnoses.detach().cpu().numpy())

            # Update accumulated values
            total_loss += loss.detach().cpu().numpy()
            n_entries += len(traces)

        # Update progress bar
        eval_pbar.set_postfix({'loss': total_loss / n_entries})
    eval_pbar.close()
    return total_loss / n_entries, np.vstack(valid_pred), np.vstack(valid_true)

def filter_adversarial(ecg_sample_adv, sample_rate, fc=100):
    fc = fc # [Hz], cutoff frequency
    fst = fc*1.1  # [Hz], rejection band
    rp = 0.5  # [dB], ripple in passband
    rs = 40  # [dB], attenuation in rejection band
    wn = fc / (sample_rate / 2)
    wst = fst / (sample_rate / 2)

    filterorder, aux = sgn.ellipord(wn, wst, rp, rs)
    sos = sgn.iirfilter(filterorder, wn, rp, rs, btype='low', ftype='ellip', output='sos')
    ecg_sample_adv = sgn.sosfiltfilt(sos, ecg_sample_adv, padtype='constant', axis=-1)
    return ecg_sample_adv