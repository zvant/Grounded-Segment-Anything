import argparse
import os
import sys
import gc
import tqdm
import random
import math

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), 'GroundingDINO'))
sys.path.append(os.path.join(os.getcwd(), 'segment_anything'))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from torchvision.ops.boxes import batched_nms

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor,
)
from segment_anything.utils.amg import batched_mask_to_box, remove_small_regions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert('RGB')  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device='cpu'):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f'({str(logit.max().item())[:4]})')
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches='tight', dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def detect_from_dino(args):
    import contextlib
    from detectron2.structures import BoxMode
    sys.path.append(os.path.join(args.s100dir, 'scripts'))
    from evaluation import evaluate_masked

    if args.id == 'batch':
        video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
    else:
        video_id_list = [args.id]
    text_prompt_0 = 'person . people . pedestrian . driver .'
    text_prompt_1 = 'vehicle . car . suv . bus . truck .'

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    # sam_hq_checkpoint = args.sam_hq_checkpoint
    # use_sam_hq = args.use_sam_hq
    # image_path = args.input_image
    # text_prompt = args.text_prompt
    # output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    # device = args.device
    print('prompt: %s / %s, box threshold: %f, text threshold: %f' % (text_prompt_0, text_prompt_1, box_threshold, text_threshold))
    text_prompt_list = [text_prompt_0, text_prompt_1]

    # make dir
    # os.makedirs(output_dir, exist_ok=True)
    # load image
    # load model
    model = load_model(config_file, grounded_checkpoint, device='cuda')
    model.eval()
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).cuda())

    APs = {}
    for video_id in video_id_list:
        inputdir = os.path.normpath(os.path.join(args.s100dir, 'images', 'annotated', video_id))
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        detections = []
        with torch.no_grad():
            for im in tqdm.tqdm(images, ascii=True, desc=video_id):
                # torch.cuda.empty_cache()
                im['annotations'] = []
                f = os.path.join(inputdir, 'unmasked', im['file_name'])
                image_pil, im_dino = load_image(f)
                im_sam = cv2.imread(f)
                im_sam = cv2.cvtColor(im_sam, cv2.COLOR_BGR2RGB)
                predictor.set_image(im_sam)
                # im_sam_half = cv2.resize(im_sam, [im_sam.shape[1] // 2, im_sam.shape[0] // 2], cv2.INTER_LINEAR)
                # predictor.set_image(im_sam_half)
                size = image_pil.size
                H, W = size[1], size[0]

                boxes_filt_classes, pred_phrases_classes = [], []
                for c in range(0, len(text_prompt_list)):
                    boxes_filt, pred_phrases = get_grounding_output(model, im_dino, text_prompt_list[c], box_threshold, text_threshold, device='cuda')
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt_classes.append(boxes_filt.cpu())
                    pred_phrases_classes.extend([(c, p) for p in pred_phrases])

                boxes_filt_classes = torch.cat(boxes_filt_classes, dim=0)
                if boxes_filt_classes.size(0) < 1:
                    continue
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt_classes, im_sam.shape[:2]).cuda()
                # transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt_classes / 2, im_sam_half.shape[:2]).cuda()

                masks = []
                box_bs = 200
                for b in range(0, math.ceil(boxes_filt_classes.size(0) / box_bs)):
                    _boxes_batch = transformed_boxes[b * box_bs : (b + 1) * box_bs]
                    if _boxes_batch.size(0) < 1:
                        continue
                    masks.append(
                        predictor.predict_torch(
                            point_coords = None,
                            point_labels = None,
                            boxes = _boxes_batch,
                            multimask_output = False,
                        )[0].cpu().numpy()
                    )
                masks = np.concatenate(masks, axis=0)
                for m, (c, lb) in zip(masks, pred_phrases_classes):
                    xs, ys = np.where(m[0].T)
                    # xs, ys = xs * 2, ys * 2
                    if xs.shape[0] > 4:
                        im['annotations'].append({'bbox': list(map(float, [xs.min(), ys.min(), xs.max(), ys.max()])), 'segmentation': [], 'category_id': c, 'score': float(lb[lb.find('(') + 1 : lb.find(')')]), 'bbox_mode': BoxMode.XYXY_ABS})
                detections.append(im)

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        del APs[video_id]['raw']
        print(video_id, APs[video_id]['results'])

    categories = ['person', 'vehicle', 'overall', 'weighted']
    print('videos average:')
    for c in categories:
        _AP_videos = np.array([APs[v]['results'][c] for v in APs])
        print(c, _AP_videos.mean(axis=0))
    with open(os.path.join(os.path.dirname(__file__), 'scenes100_APs_%s.json' % args.sam_version), 'w') as fp:
        json.dump(APs, fp)


def detect_from_kde(args):
    from matplotlib.backends.backend_pdf import PdfPages

    def _add_boxes(ax, boxes, scores, thres=0.5):
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s < thres:
                continue
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    def _get_image_masks(masks, scores, thres=0.5):
        masks = sorted(list(masks), key=lambda m: m.sum() * -1)
        image = np.zeros((masks[0].shape[0], masks[0].shape[1], 4))
        for m, s in zip(masks, scores):
            if s >= thres:
                image[m] = np.concatenate([np.random.random(3), [0.5]])
        return image

    with open(os.path.join(os.path.dirname(__file__), 'GroundingDINO', 'scenes100_detections_all_swint.json'), 'r') as fp:
        detections_all = json.load(fp)

    predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).cuda())
    num_queries = 200
    for video_id, detections in detections_all.items():
        # if video_id <= '160':
        #     continue
        npz = np.load(os.path.join(os.path.dirname(__file__), 'GroundingDINO', 'KDE_thres', '%s.b3.00.npz' % video_id))
        saliency = npz['saliency']
        npz.close()
        points_classes = []
        for k in range(0, saliency.shape[0]):
            points, stride = [], 16
            for x in range(stride, saliency.shape[2], stride):
                for y in range(stride, saliency.shape[1], stride):
                    points.append([x, y, saliency[k, y - stride : y + stride, x - stride : x + stride].mean()])
            random.shuffle(points)
            points_classes.append(np.array(sorted(points, key=lambda t: t[2] * -1))[:num_queries])
            # points_classes.append(np.array([[10, 10, 1], [1900, 1060, 1]]))

        inputdir = os.path.normpath(os.path.join(args.s100dir, 'images', 'annotated', video_id))
        detections_sam = []
        if len(detections) > 8:
            detections = detections[::len(detections) // 8]
        for im in tqdm.tqdm(detections, ascii=True, desc=video_id):
            im_sam = cv2.imread(os.path.join(inputdir, 'unmasked', im['file_name']))
            im_sam = cv2.cvtColor(im_sam, cv2.COLOR_BGR2RGB)
            predictor.set_image(im_sam)
            results_classes = []
            for k in range(0, len(points_classes)):
                point_coords = predictor.transform.apply_coords(points_classes[k][:, :2], im_sam.shape[:2])
                masks, mask_scores, _ = predictor.predict_torch(
                    point_coords = torch.from_numpy(point_coords).unsqueeze(1).cuda(),
                    point_labels = torch.ones(points_classes[k].shape[0], dtype=torch.long).unsqueeze(1).cuda(),
                    boxes = None,
                    multimask_output = False,
                )
                masks, mask_scores = masks[:, 0, :, :], mask_scores[:, 0]
                for i in range(0, masks.size(0)):
                    _m = remove_small_regions(masks[i].cpu().numpy(), 16, mode='holes')[0]
                    masks[i] = torch.from_numpy(remove_small_regions(_m, 16, mode='islands')[0])
                boxes = batched_mask_to_box(masks)
                # print(boxes.min(dim=0), boxes.max(dim=0))
                keep_by_nms = batched_nms(boxes.float(), mask_scores, torch.zeros_like(boxes[:, 0]), iou_threshold=0.7)
                # print(masks.size(), mask_scores.size(), mask_scores.min(), mask_scores.max(), mask_scores.mean(), boxes.size(), keep_by_nms.size())
                masks, boxes, mask_scores = masks[keep_by_nms], boxes[keep_by_nms], mask_scores[keep_by_nms]
                results_classes.append({'masks': _get_image_masks(masks.cpu().numpy(), mask_scores.cpu().numpy(), thres=0), 'boxes': boxes.cpu().numpy(), 'scores': mask_scores.cpu().numpy()})
            detections_sam.append({'image': im_sam, 'results': results_classes})

        with PdfPages(os.path.join(os.path.dirname(__file__), 'GroundingDINO', 'KDE_thres', 'sam_%s.pdf' % video_id)) as pdf:
            for i in range(0, len(detections_sam)):
                _, axes = plt.subplots(3, 2, figsize=(16, 14))
                axes[0][0].imshow(saliency[0], cmap='gray')
                axes[0][0].set_title('person KDE')
                axes[0][1].imshow(saliency[1], cmap='gray')
                axes[0][1].set_title('vehicle KDE')

                for k in range(0, len(points_classes)):
                    axes[1][k].imshow(detections_sam[i]['image'])
                    axes[1][k].scatter(points_classes[k][:, 0], points_classes[k][:, 1], marker='x', c='r')
                    axes[1][k].set_xlim(0, saliency.shape[2])
                    axes[1][k].set_ylim(saliency.shape[1], 0)
                    axes[1][k].set_title('%d prompt points' % points_classes[k].shape[0])

                    axes[2][k].imshow(detections_sam[i]['image'].mean(axis=2), cmap='gray')
                    axes[2][k].imshow(detections_sam[i]['results'][k]['masks'])
                    axes[2][k].set_title('%s masks' % detections_sam[i]['results'][k]['scores'].shape[0])
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        del detections_sam, pdf
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grounded-Segment-Anything Demo', add_help=True)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--grounded_checkpoint', type=str, help='path to checkpoint file')
    parser.add_argument('--sam_version', type=str, default='vit_h', help='SAM ViT version: vit_b / vit_l / vit_h')
    parser.add_argument('--sam_checkpoint', type=str, help='path to sam checkpoint file')
    # parser.add_argument('--sam_hq_checkpoint', type=str, default=None, help='path to sam-hq checkpoint file')
    # parser.add_argument('--use_sam_hq', action='store_true', help='using sam-hq for prediction')
    # parser.add_argument('--input_image', type=str, required=True, help='path to image file')
    # parser.add_argument('--text_prompt', type=str, required=True, help='text prompt')
    # parser.add_argument('--output_dir', '-o', type=str, default='outputs', required=True, help='output directory')

    parser.add_argument('--box_threshold', type=float, default=0.05, help='box threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='text threshold')
    parser.add_argument('--id', type=str, default='batch')
    parser.add_argument('--s100dir', type=str, default='')
    # parser.add_argument('--device', type=str, default='cpu', help='running on cpu only!, default=False')
    args = parser.parse_args()
    if args.opt == 'detect':
        detect_from_dino(args)
    if args.opt == 'kde_sample':
        detect_from_kde(args)


'''
python inference_SAM_scenes100.py --opt detect --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint GroundingDINO/groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --id batch --s100dir ../Intersections
python inference_SAM_scenes100.py --opt detect --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint GroundingDINO/groundingdino_swint_ogc.pth --sam_version vit_l --sam_checkpoint sam_vit_l_0b3195.pth --id batch --s100dir ../Intersections
python inference_SAM_scenes100.py --opt detect --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint GroundingDINO/groundingdino_swint_ogc.pth --sam_version vit_b --sam_checkpoint sam_vit_b_01ec64.pth --id batch --s100dir ../Intersections

GroundingDINO swint
SAM ViT-L
videos average:
person [0.3542701  0.57535193]
vehicle [0.37942078 0.55589336]
overall [0.46816129 0.68357949]
weighted [0.48534837 0.71429242]

GroundingDINO swint
SAM ViT-B
videos average:
person [0.34871872 0.57441905]
vehicle [0.37024638 0.55418873]
overall [0.46001884 0.68202343]
weighted [0.47567415 0.71278502]

python inference_SAM_scenes100.py --opt kde_sample --sam_version vit_l --sam_checkpoint sam_vit_l_0b3195.pth --s100dir ../Intersections
'''
