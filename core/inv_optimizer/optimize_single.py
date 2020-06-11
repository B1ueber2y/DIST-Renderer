import os, sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.utils.loss_utils import *
from core.utils.render_utils import *
from core.visualize.visualizer import print_loss_pack, Visualizer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loss_single import compute_all_loss

'''
This function performs inverse optimization over either shape or camera parameters on single-view 2D inputs.

Required parameters:
- sdfrenderer_list:     A list of differentiable sdf renderer to support multiscale optimization. If multiscale is not enabled, a list containing one single renderer is needed.
- evaluator:            An instance of class <"Evaluator">
- optimizer:            An instance of PyTorch optimizer, either for shape code or camera extrinsics, which is optional for the key "optimizer_type".
- shape_code:           class <torch.Tensor>
- camera_tensor:        If "optimizer_type" is set to "shape", this should be a extrinsic parameter tensor (sized [3, 4]).
                        If "optimizer_type" is set to "camera", this should be a camera tensor (sized [7], containing quaternion and translation). Please refer to "get_camera_from_tensor" and "get_tensor_from_camera" functions in core/utils/render_utils.py.
- gt_pack:              A dictionary containing the groundtruth targets: {"depth": gt_depth, "normal": gt_normal, "silhouette": gt_silhouette}. If not applicable, None is passed to the value.
- weight_dict:          A dictionary of weights.

Optional parameters:
- optimizer_type:       "shape" or "camera", defining the target that the optimization is performed over.
- num_iters:            number of iterations.
- renderer_weights:     A list defining different weights for different renderer under the multiscale setting. If [] is passed it means that all renderers are equally treated.
- grad_settings:        A dictionary on whether to have gradients for different types of groundtruth.
- points_gt:            groundtruth shapes for evaluation.
- test_step:            evaluation interval.

Returns:
If optimizer_type is set to "shape", shape_code and optimizer is returned.
If optimizer_type is set to "camera", camera_tensor and optmizer is returned.
'''
def optimize_single_view(sdfrenderer_list, evaluator, optimizer, shape_code, camera_tensor, gt_pack, weight_dict, optimizer_type='shape', num_iters=200,
                        renderer_weights=[], grad_settings={"depth": True, "normal": True, "silhouette": True},
                        points_gt=None, test_step=50, profile=False, visualizer=None, silent=False, vis_folder=None, ray_marching_type='pyramid_recursive'):

    # initialize workspace
    print_flag = (not silent)
    visualize_flag = (visualizer is not None) and (not silent)
    if visualize_flag:
        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)
    if renderer_weights == []:
        for idx in range(len(sdfrenderer_list)):
            renderer_weights.append(1.0)

    # optimization
    for i in tqdm(range(num_iters)):
        optimizer.zero_grad()
        if optimizer_type == 'shape':
            extrinsics = camera_tensor
        elif optimizer_type == 'camera':
            extrinsics = get_camera_from_tensor(camera_tensor)
        else:
            raise NotImplementedError

        loss = 0
        for renderer_idx, (renderer, renderer_weight) in enumerate(zip(sdfrenderer_list, renderer_weights)):
            if renderer_idx == 0:
                visualizer_ = visualizer
            else:
                visualizer_ = None
            loss_pack, visualizer_ = compute_all_loss(renderer, shape_code, extrinsics, gt_pack, threshold=renderer.get_threshold(), profile=profile, visualizer=visualizer_, ray_marching_type=ray_marching_type, grad_settings=grad_settings)
            # visualize the original resolution if needed
            if renderer_idx == 0:
                visualizer = visualizer_
                if print_flag:
                    print_loss_pack(loss_pack, '{0}/s224'.format(i))
                if visualize_flag:
                    visualizer.show_loss_curve(os.path.join(vis_folder, 'vis_loss_curve_{}.png'.format(i)))
                    visualizer.show_all_data(os.path.join(vis_folder, 'vis_all_data_{}.png'.format(i)))
            loss_now = weight_dict['w_depth'] * loss_pack['depth'] + \
                weight_dict['w_normal'] * loss_pack['normal'] + \
                weight_dict['w_mask_gt'] * loss_pack['mask_gt'] + \
                weight_dict['w_mask_out'] * loss_pack['mask_out'] + \
                weight_dict['w_l2reg'] * loss_pack['l2reg']
            loss = loss + loss_now * renderer_weight

        if visualize_flag:
            visualizer.add_loss(loss)
        loss.backward()
        optimizer.step()

        # evaluation
        if (points_gt is not None) and ((i+1) % test_step == 0) and (not silent):
            fname = os.path.join(vis_folder, 'output_{}.ply'.format(i))
            points_tmp = evaluator.latent_vec_to_points(shape_code, fname=fname, silent=True)
            if points_tmp is None:
                print('The current latent code does not correspond to a valid shape.')
                dist = 1e11
            else:
                dist = evaluator.compute_chamfer_distance(points_gt, points_tmp)
                print('CHAMFER DISTANCE: {0:.3f}'.format(dist * 1000))
            if visualize_flag:
                visualizer.add_chamfer(dist)

        # dump data
        if visualize_flag:
            visualizer.dump_all_data(os.path.join(vis_folder, 'vis_all_data_{}.pkl'.format(i)))
    # return
    if optimizer_type == 'shape':
        return shape_code, optimizer
    elif optimizer_type == 'camera':
        return camera_tensor, optimizer
    else:
        raise NotImplementedError

