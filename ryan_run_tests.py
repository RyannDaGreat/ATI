from rp import *

set_current_directory(get_absolute_path('~/CleanCode/Github/ATI'))

if input_yes_no('Rebuild test dataset?'):
    #Wipe and start from scratch...
    r._run_sys_command(f'rm -rf ryan_examples ; {sys.executable} ryan_convert_gaussblob_to_ati_examples.py')

command = """export CUDA_VISIBLE_DEVICES=%i && echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" && bash run_example.sh -p ryan_examples/test.yaml -c /home/jupyter/CleanCode/Huggingface/Wan2.1-ATI-14B-480P/Wan2.1-ATI-14B-480P -o ryan_samples"""#bash

session = "ATI_INFER"

yaml = tmuxp_create_session_yaml(
    {"INFER": [command%i for i in get_all_gpu_ids()]},
    session_name=session,
    command_before="cd ~CleanCode/Github/ATI",
)

tmux_kill_session(session)
tmuxp_launch_session_from_yaml(yaml)
