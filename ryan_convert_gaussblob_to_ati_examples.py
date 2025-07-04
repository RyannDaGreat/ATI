
import ryan_tracks_converter as rtc
import torch

import rp
from rp import *

samples_dirs = [
    "/home/jupyter/CleanCode/Github/DaS_Trees/gauss_blobs/source/editor/untracked/inferblobs_edit_results",
    "/home/jupyter/CleanCode/Github/DaS_Trees/gauss_blobs/source/editor/untracked/inferblobs_test_results",
]
examples_root="/home/jupyter/CleanCode/Github/ATI/ryan_examples"
assert rp.get_current_directory() == "/home/jupyter/CleanCode/Github/ATI"


@rp.squelch_wrap
def sample_dir_hash(sample_dir):
    return tuple(
        rp.get_sha256_hash(rp.file_to_bytes(x))
        for x in rp.path_join(sample_dir, ["target_tracks.pth", "counter_video.mp4"])
    )
sample_dirs = rp.list_flatten([rp.get_subfolders(samples_dir) for samples_dir in samples_dirs])
sample_dirs = rp.unique(sample_dirs, key=sample_dir_hash)

for sample_dir in rp.eta(sample_dirs):
    try:
        title = rp.get_folder_name(sample_dir)
        safe_title = title.replace(" ", "_").replace("[", "").replace("]", "--")

        metadata = rp.load_json(rp.path_join(sample_dir, "metadata.json"))
        prompt = metadata.prompt

        examples_root = "ryan_examples"
        first_frame_path = f"{examples_root}/images/{safe_title}.jpg"
        tracks_path = f"{examples_root}/tracks/{safe_title}.pth"
        yaml_path = f"{examples_root}/test.yaml"
        #
        new_yaml_lines = rp.line_join(
            f"- image: {repr(first_frame_path)} #Make local to repo root",
            f"  text: {repr(prompt)}",
            f"  track: {repr(tracks_path)}",
            f"  title: {repr(title)}",
        )
        #
        if file_exists(yaml_path) and new_yaml_lines in rp.load_text_file(yaml_path):
            continue


        visibles = torch.load(rp.path_join(sample_dir, "target_visibles.pth"),map_location='cpu')
        tracks = torch.load(rp.path_join(sample_dir, "target_tracks.pth"),map_location='cpu')
        #                    ┌                                                      ┐
        #                    │┌                                             ┐       │
        #                    ││                ┌          ┐   ┌            ┐│       │
        tracks = torch.concat([tracks, visibles[:, :, None].to(tracks.dtype)], dim=2)  # T N XY form
        #                    ││                └          ┘   └            ┘│       │
        #                    │└                                             ┘       │
        #                    └                                                      ┘
        tracks *= 8  # Not sure why but they scale it by 8...
        tracks = tracks.to(torch.float32)

        first_frame = next(rp.load_video_stream(rp.path_join(sample_dir, "counter_video.mp4")))
        
        #Scale it spatially
        #old_height,old_width=get_image_dimensions(first_frame)
        #new_height,new_width=480,832 #ATI needs these dimensions
        #first_frame=cv_resize_image(first_frame,(new_height,new_width))
        #tracks[:,0]*=new_width/old_width
        #tracks[:,1]*=new_height/old_height
        
        tracks=resize_list(tracks, 121)

        rp.make_parent_directory(tracks_path)
        rp.make_parent_directory(first_frame_path)

        rp.save_image(first_frame, first_frame_path)
        rtc.save_weird_tracks_file(tracks, tracks_path)
        rp.append_line_to_file(new_yaml_lines,yaml_path)
        
    except Exception as e:
        rp.fansi_print(e,'red bold')

