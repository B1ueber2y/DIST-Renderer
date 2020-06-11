'''
Ported from [DeepSDF, CVPR'19]
https://github.com/facebookresearch/DeepSDF
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"

def get_model_params_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, model_params_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir

def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, optimizer_params_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir

def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):
    dir = os.path.join(experiment_dir, latent_codes_subdir)
    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)
    return dir

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))

def get_learning_rate_schedules(specs):
        schedule_specs = specs["LearningRateSchedule"]
        schedules = []
        for schedule_specs in schedule_specs:
            if schedule_specs["Type"] == "Step":
                schedules.append(
                    StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )
            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )
        return schedules

def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

def latent_size_regul(latent, indices):
    latent_loss = 0.0
    for ind in indices:
        latent_loss += torch.mean(latent[ind].pow(2))
    return latent_loss / len(indices)

def save_model(experiment_directory, filename, model, epoch):
    model_params_dir = get_model_params_dir(experiment_directory, True)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(model_params_dir, filename),
    )

def load_model(experiment_directory, filename, model):
    model_params_dir = get_model_params_dir(experiment_directory)
    data = torch.load(os.path.join(model_params_dir, filename))
    model.load_state_dict(data['model_state_dict'])
    return model, data['epoch']

def save_optimizer(experiment_directory, filename, optimizer, epoch):
    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)
    optimizer.load_state_dict(data["optimizer_state_dict"])
    return optimizer, data['epoch']

def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):

        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])
    return lat_vecs, data["epoch"]

# function from [PMO, CVPR'19]
# https://github.com/chenhsuanlin/photometric-mesh-optim/blob/0fa3d50f4e9e11ed4fdbe18590a47e3b242ce279/pose.py#L5
def params_to_mtrx(sim3):
    R = get_lie_rotation_matrix(sim3['rot'])
    mtrx = torch.cat([sim3['scale'].exp()*R,sim3['trans'][:,None]],dim=1)
    return mtrx

# function from [PMO, CVPR'19]
# https://github.com/chenhsuanlin/photometric-mesh-optim/blob/0fa3d50f4e9e11ed4fdbe18590a47e3b242ce279/pose.py#L10
def get_lie_rotation_matrix(r):
    O = torch.tensor(0.0,dtype=torch.float32, device='cuda')
    rx = torch.stack([torch.stack([O,-r[2],r[1]]),
                      torch.stack([r[2],O,-r[0]]),
                      torch.stack([-r[1],r[0],O])],dim=0)
    # matrix exponential
    R = torch.eye(3, dtype=torch.float32, device='cuda')
    numer = torch.eye(3, dtype=torch.float32, device='cuda')
    denom = 1.0
    for i in range(1,20):
        numer = numer.matmul(rx)
        denom *= i
        R += numer/denom
    return R
