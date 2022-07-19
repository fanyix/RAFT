
###############################################################################
## Batch process datasets
###############################################################################

import sys
sys.path.append('core')

import argparse
import os
import os.path as osp
import cv2
import numpy as np
import torch
import shutil
import time

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import kornia
from moviepy.editor import VideoFileClip, ImageSequenceClip


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
parser.add_argument('--part',
                    dest='part',
                    type=int,
                    default=0,
                    help='part index for the job')
parser.add_argument('--parts',
                    dest='parts',
                    type=int,
                    default=1,
                    help='total number of parts')
args = parser.parse_args()


# Parameters.
args.device = 'cuda' # 'cuda', 'cpu'
args.model_path = '../data/models/raft-things.pth' # raft-things.pth
args.video_list_path = 'TODO'
args.video_dir = 'TODO'
args.output_dir = 'TODO'
args.resize_ratio = 1.0
args.gap = 5 # 5
args.concat_vis = False # False
args.mode = 'flow_edge' # 'flow_edge_selective', 'flow_edge', 'flow_vgg'
args.flow_range_min = 0.0 # 0.0, -20.0
args.flow_range_max = 10.0 # 10.0, 20.0
args.select_thresh = 0.02 # 0.02
args.frame_thresh = 90  # 90
args.break_token = 'stop'


def process_frame(frame, prev_frames, args):
    orig_frame = frame
    H, W, C = frame.shape
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)[None]
    frame = frame.float().to(args.device)

    if len(prev_frames) > 0:
        
        prev = prev_frames[0]
        padder = InputPadder(prev.shape)
        prev, output = padder.pad(prev, frame)
        # flow_low, flow_up = model(prev, output, iters=20, test_mode=True)
        flow_low, flow_up = model(output, prev, iters=20, test_mode=True)
        flow_up = padder.unpad(flow_up)
        
        if args.mode == 'flow_mag':
            
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_up = np.linalg.norm(flow_up, axis=2)
            flow_up = (flow_up - np.min(flow_up)) / (np.max(flow_up) - np.min(flow_up) + 1e-8) * 255
            output = flow_up[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
        
        elif args.mode == 'flow_mag_thresh':
            
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_up = np.linalg.norm(flow_up, axis=2)
            # print('mean, std, min, max = {}, {}, {}, {}'.format(np.mean(flow_up), np.std(flow_up), np.min(flow_up), np.max(flow_up)))
            flow_up = np.minimum(np.maximum(flow_up - args.flow_range_min, 0.0) / (args.flow_range_max - args.flow_range_min), 1.0) * 255
            output = flow_up[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
        
        elif args.mode == 'flow_mag_selective':
            
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_up = np.linalg.norm(flow_up, axis=2)
            flow_up_edge = kornia.filters.Sobel()(torch.from_numpy(flow_up)[None, None])
            flow_up_edge = flow_up_edge.squeeze().numpy()
            if np.sum(flow_up_edge) / flow_up_edge.size > args.select_thresh:
                flow_up = np.minimum(np.maximum(flow_up - args.flow_range_min, 0.0) / (args.flow_range_max - args.flow_range_min), 1.0) * 255
                output = flow_up[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
            else:
                output = None
        
        elif args.mode == 'flow_edge':
            
            # # This is for ploting ICCV motion inputs figure 
            # flow_vis = flow_viz.flow_to_image(flow_up[0].permute(1, 2, 0).cpu().numpy())
            # prev_frame = padder.pad(prev_frames[-1], frame)[0]
            # frame_diff = output.float() - prev_frame.float()
            # frame_diff = torch.mean(torch.abs(frame_diff), dim=1, keepdim=True)
            # frame_diff = frame_diff.repeat(1, 3, 1, 1)
            # frame_diff = frame_diff.permute(0, 2, 3, 1).cpu().numpy()
            # frame_diff = frame_diff[0]
            # frame_diff = frame_diff.astype(np.uint8)

            # flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # flow_up = np.linalg.norm(flow_up, axis=2)
            # flow_up = kornia.filters.Sobel()(torch.from_numpy(flow_up)[None, None])
            # flow_up = flow_up.squeeze().numpy()
            # flow_up = np.minimum(np.maximum(flow_up - args.flow_range_min, 0.0) / (args.flow_range_max - args.flow_range_min), 1.0) * 255
            # output = flow_up[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
            flow_up = torch.norm(flow_up, dim=1, keepdim=True)
            flow_up_edge = kornia.filters.Sobel()(flow_up)
            flow_up_edge = flow_up_edge.squeeze()
            flow_up_edge = (torch.clamp(flow_up_edge, min=args.flow_range_min, max=args.flow_range_max) - args.flow_range_min) / (args.flow_range_max - args.flow_range_min + 1e-8) * 255
            output = flow_up_edge.cpu().numpy()[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
        
        elif args.mode == 'flow_edge_selective':
            
            flow_up = torch.norm(flow_up, dim=1, keepdim=True)
            flow_up_edge = kornia.filters.Sobel()(flow_up)
            if torch.sum(flow_up_edge) / flow_up_edge.nelement() > args.select_thresh:
                flow_up_edge = flow_up_edge.squeeze()
                flow_up_edge = (torch.clamp(flow_up_edge, min=args.flow_range_min, max=args.flow_range_max) - args.flow_range_min) / (args.flow_range_max - args.flow_range_min + 1e-8) * 255
                output = flow_up_edge.cpu().numpy()[:, :, np.newaxis].repeat(3, axis=2).astype(np.uint8)
            else:
                output = None
        
        elif args.mode == 'flow_map':
            
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            output = flow_viz.flow_to_image(flow_up)
        
        elif args.mode == 'flow_vgg':
            
            flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_up = np.minimum(np.maximum(flow_up - args.flow_range_min, 0.0) / (args.flow_range_max - args.flow_range_min), 1.0) * 255
            flow_up = np.concatenate([flow_up, np.zeros((flow_up.shape[0], flow_up.shape[1], 1))], axis=2)
            output = flow_up.astype(np.uint8)
        
        else:
            raise RuntimeError('Unknown mode.')
        
        if output is not None:
            output = cv2.resize(output, (int(W * args.resize_ratio), int(H * args.resize_ratio)))
            if args.concat_vis:
                orig_frame = cv2.resize(orig_frame, (int(W * args.resize_ratio), int(H * args.resize_ratio)))
                output = np.concatenate([orig_frame, output], axis=0)

            # # This is for ploting ICCV motion inputs figure  
            # flow_vis = cv2.resize(flow_vis, (int(W * args.resize_ratio), int(H * args.resize_ratio)))
            # frame_diff = cv2.resize(frame_diff, (int(W * args.resize_ratio), int(H * args.resize_ratio)))
            # output = np.concatenate([output, flow_vis, frame_diff], axis=0)

    else:
        
        if args.concat_vis:
            output = np.zeros((int(H * args.resize_ratio) * 2, int(W * args.resize_ratio), C))
            
            # # This is for ploting ICCV motion inputs figure 
            # output = np.zeros((int(H * args.resize_ratio) * 4, int(W * args.resize_ratio), C))

        else:
            output = np.zeros((int(H * args.resize_ratio), int(W * args.resize_ratio), C))
        output = output.astype(np.uint8)
    
    prev_frames.append(frame)
    if len(prev_frames) > args.gap:
        prev_frames.pop(0)
    return output, prev_frames
    

def iterate_video_selective(input_path, output_path, model, args):
    video = VideoFileClip(input_path)
    fps = video.fps
    frames = video.iter_frames()
    prev_frames, outputs = [], []
    with torch.no_grad():
        for frame in frames:
            output, prev_frames = process_frame(frame, prev_frames, args)
            if output is not None:
                outputs.append(output)
    video.close()
    if len(outputs) >= args.frame_thresh:
        out_video = ImageSequenceClip(outputs, fps=fps) 
        out_video.write_videofile(output_path + '_tmp.mp4', logger=None)
        while not osp.isfile(output_path + '_tmp.mp4'):
            print('Wait for {} to be ready.'.format(output_path + '_tmp.mp4'))
            time.sleep(1)
        shutil.move(output_path + '_tmp.mp4', output_path)


def iterate_video(input_path, output_path, model, args):
    # One behavior I found for MovieClip is that for a N-frame video, it will generate N+1 frames, 
    # where the last generated frame (i.e., N+1-th frame) is a duplicate of the N-th frame.  
    global prev_frames
    prev_frames = []
    video = VideoFileClip(input_path)
    def transform_frames(get_frame, t):
        global prev_frames
        frame = get_frame(t)
        output, prev_frames = process_frame(frame, prev_frames, args)    
        return output
    with torch.no_grad():
        video = video.fl(transform_frames, keep_duration=True, apply_to='mask')
        video.write_videofile(output_path + '_tmp.mp4', audio=False, logger=None)
        video.close()
    while not osp.isfile(output_path + '_tmp.mp4'):
        print('Wait for {} to be ready.'.format(output_path + '_tmp.mp4'))
        time.sleep(1)
    shutil.move(output_path + '_tmp.mp4', output_path)


def generate_flow_video(input_path, output_path, model, args):
    if 'selective' in args.mode:
        iterate_video_selective(input_path, output_path, model, args)
    else:
        iterate_video(input_path, output_path, model, args)


# Read video list.
with open(args.video_list_path, 'r') as h:
    video_list = h.readlines()
video_list = [x.rstrip('\n').split(' ')[0] for x in video_list]

# Initialize the model.
model = torch.nn.DataParallel(RAFT(args))
if args.device == 'cpu':
    loaded_model = torch.load(args.model_path, map_location=torch.device('cpu'))
else:
    loaded_model = torch.load(args.model_path)
model.load_state_dict(loaded_model)
model = model.module
model.to(args.device)
model.eval()

# Shuffle the video list.
# np.random.shuffle(video_list)

# Loop over videos.
for video_idx in range(args.part, len(video_list), args.parts):
    video = video_list[video_idx]
    video_path = osp.join(args.video_dir, video)
    output_path = osp.join(args.output_dir, video)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    # if not osp.isfile(output_path) and not osp.isfile(output_path + '.lock'):
    if not osp.isfile(output_path):
        try:
            # open(output_path + '.lock', 'a').close()
            generate_flow_video(video_path, output_path, model, args)
            # os.remove(output_path + '.lock')
        except Exception as e:
            print('{}, failed to process {}.'.format(e, video_path))
    print('{}/{}'.format(video_idx, len(video_list)))
    if osp.exists(args.break_token):
        print('Break token {} detected, exiting.'.format(args.break_token))
        break
