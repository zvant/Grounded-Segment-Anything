import argparse
import os
import sys
import json
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
bbox_rgbs = ['#FF0000', '#0000FF']


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


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda' if not cpu_only else 'cpu'
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith('.'):
        caption = caption + '.'
    # device = 'cuda' if not cpu_only else 'cpu'
    # model = model.to(device)
    # image = image.to(device)
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
        pred_phrases.append([pred_phrase.split(' '), float(logit.max().item())])

    return boxes_filt, pred_phrases


def detect_scenes100(args):
    import contextlib
    from detectron2.structures import BoxMode
    sys.path.append(os.path.join(args.s100dir, 'scripts'))
    from evaluation import evaluate_masked

    text_prompt_0 = 'person . people . pedestrian . driver .'
    text_prompt_1 = 'vehicle . car . suv . bus . truck .'

    # cfg
    if args.config == 'swint':
        config_file = os.path.join(os.path.dirname(__file__), 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'groundingdino_swint_ogc.pth')
    else:
        config_file = os.path.join(os.path.dirname(__file__), 'groundingdino', 'config', 'GroundingDINO_SwinB.cfg.py')
        checkpoint_path = os.path.join(os.path.dirname(__file__), 'groundingdino_swinb_cogcoor.pth')
    box_threshold = args.box_threshold
    text_threshold = args.box_threshold
    print('prompt: %s / %s, box threshold: %f, text threshold: %f' % (text_prompt_0, text_prompt_1, box_threshold, text_threshold))
    text_prompt_list = [text_prompt_0, text_prompt_1]
    model = load_model(config_file, checkpoint_path, cpu_only=False).to('cuda')
    model.eval()

    detections_all, APs_all = {}, {}
    for video_id in video_id_list:
        inputdir = os.path.normpath(os.path.join(args.s100dir, 'images', 'annotated', video_id))
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        detections = []
        with torch.no_grad():
            for im in tqdm.tqdm(images, ascii=True, desc=video_id):
                _, im_arr = load_image(os.path.join(inputdir, 'unmasked', im['file_name']))
                im['annotations'] = []
                for c in range(0, len(text_prompt_list)):
                    boxes_filt, pred_phrases = get_grounding_output(model, im_arr.to('cuda'), text_prompt_list[c], box_threshold, text_threshold, cpu_only=False)
                    for box, lb in zip(boxes_filt, pred_phrases):
                        xc, yc, w, h = map(float, [box[0] * im['width'], box[1] * im['height'], box[2] * im['width'], box[3] * im['height']])
                        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                        im['annotations'].append({'bbox': [x1, y1, x2, y2], 'segmentation': [], 'category_id': c, 'score': float(lb[1]), 'bbox_mode': BoxMode.XYXY_ABS})
                detections.append(im)

        detections_all[video_id] = detections
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            APs_all[video_id] = evaluate_masked(video_id, detections, outputfile=None)
        del APs_all[video_id]['raw']
        print(video_id, APs_all[video_id]['results'])

    categories = ['person', 'vehicle', 'overall', 'weighted']
    print('videos average:')
    for c in categories:
        _AP_videos = np.array([APs_all[v]['results'][c] for v in APs_all])
        print(c, _AP_videos.mean(axis=0))
    with open(os.path.join(os.path.dirname(__file__), 'scenes100_APs_all_%s.json' % args.config), 'w') as fp:
        json.dump(APs_all, fp)
    with open(os.path.join(os.path.dirname(__file__), 'scenes100_detections_all_%s.json' % args.config), 'w') as fp:
        json.dump(detections_all, fp)

        #     import matplotlib.patches as patches
        #     _, ax = plt.subplots()
        #     ax.imshow(Image.open(os.path.join(inputdir, 'unmasked', im['file_name'])).convert('RGB'))
        #     for ann in im['annotations']:
        #         if ann['score'] < 0.25:
        #             continue
        #         (x1, y1, x2, y2), k = ann['xyxy'], ann['category_id']
        #         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=bbox_rgbs[k], facecolor='none')
        #         ax.add_patch(rect)
        #     plt.tight_layout()
        #     plt.show()


def saliency_kde_scenes100(args):
    import contextlib
    from detectron2.structures import BoxMode
    sys.path.append(os.path.join(args.s100dir, 'scripts'))
    from evaluation import _check_overlap, evaluate_masked
    import imantics

    with open(os.path.join(os.path.dirname(__file__), 'scenes100_detections_all_swint.json'), 'r') as fp:
        detections_all = json.load(fp)
    with open(os.path.join(os.path.dirname(__file__), 'scenes100_APs_all_swint.json'), 'r') as fp:
        APs_all = json.load(fp)

    for video_id in video_id_list:
        detections = detections_all[video_id]
        W, H = detections[0]['width'], detections[0]['height']
        with open(os.path.join(args.s100dir, 'masks.json'), 'r') as fp:
            mask = json.load(fp)
        mask = {m['video']: m['polygons'] for m in mask}[video_id]
        if len(mask) > 0:
            m_arr = imantics.Annotation.from_polygons(mask, image=imantics.Image(np.zeros(shape=(H, W, 3), dtype=np.uint8)))
            m_arr = np.expand_dims(m_arr.array.astype(np.float16), 2)
        else:
            m_arr = None
        labels, boxes, scores = [], [], []
        for im in detections:
            for ann in im['annotations']:
                assert ann['bbox_mode'] == BoxMode.XYXY_ABS
                if not _check_overlap(m_arr, ann['bbox']):
                    labels.append(ann['category_id'])
                    boxes.append(ann['bbox'])
                    scores.append(ann['score'])
        labels = np.array(labels, dtype=np.int32)
        scores = np.array(scores, dtype=np.float64)
        boxes = np.array(boxes).astype(np.float64)
        # print(boxes.min(axis=0).astype(np.int32), boxes.max(axis=0).astype(np.int32))

        # gW, gH = 48, 27
        # a, b, K = 1, 64, 100
        # a, b, K = 1, 0.5, 100
        for b in [0.8, 1.6]:
            density_classes = []
            for category_id in range(0, len(bbox_rgbs)):
                _category_mask = (labels == category_id)
                if _category_mask.sum() < 10:
                    _density = np.ones(shape=(H, W), dtype=np.float64)
                    _density /= _density.sum()
                else:
                    boxes_class, scores_class = boxes[_category_mask], scores[_category_mask]
                    cx_list = (boxes_class[:, 0] + boxes_class[:, 2]) / 2
                    cy_list = (boxes_class[:, 1] + boxes_class[:, 3]) / 2
                    w_list = boxes_class[:, 2] - boxes_class[:, 0]
                    h_list = boxes_class[:, 3] - boxes_class[:, 1]

                    _grid = np.stack(np.meshgrid(np.arange(0, W), np.arange(0, H)), axis=2).astype(np.float64)
                    _grid_x, _grid_y = _grid[:, :, 0], _grid[:, :, 1]
                    _density = np.zeros_like(_grid_x)
                    for cx, cy, w, h, s in tqdm.tqdm(zip(cx_list, cy_list, w_list, h_list, scores_class), ascii=True, total=scores_class.shape[0], desc=video_id):
                        # _density += s * a * np.exp(-0.5 * ((_grid_x - cx) ** 2 / (b * w) + (_grid_y - cy) ** 2 / (b * h))) / (b * np.sqrt(w) * np.sqrt(h))
                        _density += s * np.exp(-0.5 * ((_grid_x - cx) ** 2 / (b * w) + (_grid_y - cy) ** 2 / (b * h))) / (b * np.sqrt(w) * np.sqrt(h))
                    _density /= cx_list.shape[0]
                    # _density += 1 / (K ** 2)
                    _density /= _density.sum()
                # print(_density.shape)
                density_classes.append(_density)
            np.savez_compressed(os.path.join(os.path.dirname(__file__), 'KDE', '%s.b%.2f.npz' % (video_id, b)), saliency=np.stack(density_classes, axis=0))
            # print(APs_all[video_id]['results'])
            # for im in detections:
            #     for ann in im['annotations']:
            #         assert ann['bbox_mode'] == BoxMode.XYXY_ABS
            #         x1, y1, x2, y2 = map(round, ann['bbox'])
            #         x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, W), min(y2, H)
            #         xc, yc = map(round, [(x1 + x2) / 2, (y1 + y2) / 2])
            #         ann['density_mean'] = density_classes[ann['category_id']][y1 : y2, x1 : x2].mean()
            # plt.figure()
            # for category_id in range(0, len(bbox_rgbs)):
            #     plt.subplot(1, 2, category_id + 1)
            #     _im = density_classes[category_id]
            #     _im /= _im.max()
            #     _im[:50, :20] = 1.0
            #     plt.imshow(_im)
            # plt.show()
            # exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Grounding DINO example', add_help=True)
    parser.add_argument('--opt', type=str)
    parser.add_argument('--config', '-c', type=str, choices=['swint', 'swinb'], help='path to config file')
    # parser.add_argument('--checkpoint_path', '-p', type=str, required=True, help='path to checkpoint file')
    # parser.add_argument('--image_path', '-i', type=str, required=True, help='path to image file')
    # parser.add_argument('--text_prompt', '-t', type=str, required=True, help='text prompt')
    # parser.add_argument('--output_dir', '-o', type=str, default='outputs', required=True, help='output directory')

    parser.add_argument('--box_threshold', type=float, default=0.05, help='box threshold')
    parser.add_argument('--text_threshold', type=float, default=0.25, help='text threshold')
    parser.add_argument('--id', type=str, default='batch')
    parser.add_argument('--s100dir', type=str, default='')

    # parser.add_argument('--cpu-only', action='store_true', help='running on cpu only!, default=False')
    args = parser.parse_args()

    if args.opt == 'detect':
        detect_scenes100(args)
    if args.opt == 'kde':
        saliency_kde_scenes100(args)

'''
python inference_DINO_scenes100.py --opt detect --config swint --id batch --s100dir ../../Intersections
python inference_DINO_scenes100.py --opt kde --s100dir ../../Intersections

GroundingDINO swint
videos average:
person   [0.37583721 0.57545987]
vehicle  [0.42071553 0.55962123]
overall  [0.50234934 0.68609071]
weighted [0.52140074 0.71746851]
'''
