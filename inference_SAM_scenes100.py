import argparse
import os
import sys
import tqdm

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


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Grounded-Segment-Anything Demo', add_help=True)
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--grounded_checkpoint', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--sam_version', type=str, default='vit_h', required=False, help='SAM ViT version: vit_b / vit_l / vit_h')
    parser.add_argument('--sam_checkpoint', type=str, required=False, help='path to sam checkpoint file')
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

    import contextlib
    from detectron2.structures import BoxMode
    sys.path.append(os.path.join(args.s100dir, 'scripts'))
    from evaluation import evaluate_masked

    video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
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
    # initialize SAM
    # if use_sam_hq:
    #     predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    # else:
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).cuda())

    APs = {}
    for video_id in video_id_list[:5]:
        inputdir = os.path.normpath(os.path.join(args.s100dir, 'images', 'annotated', video_id))
        with open(os.path.join(inputdir, 'annotations.json'), 'r') as fp:
            images = json.load(fp)
        detections = []
        with torch.no_grad():
            for im in tqdm.tqdm(images, ascii=True, desc=video_id):
                torch.cuda.empty_cache()
                im['annotations'] = []
                f = os.path.join(inputdir, 'unmasked', im['file_name'])
                image_pil, im_dino = load_image(f)
                im_sam = cv2.imread(f)
                im_sam = cv2.cvtColor(im_sam, cv2.COLOR_BGR2RGB)
                predictor.set_image(im_sam)
                size = image_pil.size
                H, W = size[1], size[0]

                for c in range(0, len(text_prompt_list)):
                    boxes_filt, pred_phrases = get_grounding_output(model, im_dino, text_prompt_list[c], box_threshold, text_threshold, device='cuda')
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, im_sam.shape[:2]).cuda()

                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.cuda(),
                        multimask_output = False,
                    )
                    for m, lb in zip(masks, pred_phrases):
                        xs, ys = np.where(m[0].t().cpu().numpy())
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
    with open(os.path.join(os.path.dirname(__file__), 'scenes100_APs_%s.json' % args.config), 'w') as fp:
        json.dump(APs, fp)


'''
python inference_SAM_scenes100.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint GroundingDINO/groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --id batch --s100dir ../Intersections
python inference_SAM_scenes100.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint GroundingDINO/groundingdino_swint_ogc.pth --sam_version vit_l --sam_checkpoint sam_vit_l_0b3195.pth --id batch --s100dir ../Intersections
python inference_SAM_scenes100.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint GroundingDINO/groundingdino_swint_ogc.pth --sam_version vit_b --sam_checkpoint sam_vit_b_01ec64.pth --id batch --s100dir ../Intersections

GroundingDINO swint
SAM ViT-B

'''
