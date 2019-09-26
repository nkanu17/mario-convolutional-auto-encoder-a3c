import argparse

def train_arguments():
       
    # these are global arguments that are preset or can be set in the terminal/cmd    
    parser = argparse.ArgumentParser(description = 'A3C Super Mario')
    

    parser.add_argument('--env_name', default='SuperMarioBros-1-2-0', help='environment to train on (default: SuperMarioBros-1-4-v0)')
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--action_type", type=str, default="complex")
   
    parser.add_argument('--lr', type=float, default=1e-5, help = 'learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.00, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
   
   
    parser.add_argument("--num_processes", type=int, default=2, help='number of training processes to use')
    parser.add_argument("--num_steps", type=int, default=50, help='number of forward steps in A3C')
    parser.add_argument("--max_episodes", type=int, default=100000, help = 'maximum number of episodes')
    parser.add_argument("--max_steps", type=int, default=5000000, help = 'maximum number of episodes')
   
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    
    parser.add_argument('--pretrained_model_weights_path', default='src/pretrained/trained_cae/trained_cae_weights_grayscale.pth', help='')
    parser.add_argument('--pretrained_model_path', default='src/pretrained/trained_cae/trained_cae_model_gray_scale', help='')
    parser.add_argument("--sum_path", type=str, default="src/summary/a3c_mario")
    #saved path
    parser.add_argument("--trained_models_path", type=str, default="src/trained_models")
   
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
   
    parser.add_argument('--no_shared', default=False, help='use an optimizer without shared momentum.')
    parser.add_argument('--save_interval', type=int, default=500, help='Steps between saving')
    #parser.add_argument('--save_path',default=SAVEPATH, help='model save interval (default: {})'.format(SAVEPATH))
    parser.add_argument('--num_non_sample', type=int,default=2, help='number of non sampling processes (default: 2)')
    #parser.add_argument("--save_path", type=str, default="trained_models")

    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--new_stage",type=bool, default=False)
    args = parser.parse_args()
   
    return args

def test_arguments():
    parser = argparse.ArgumentParser(
        "Mario")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--stage", type=int, default=1	)
    parser.add_argument("--file_num", type=int, default=1)

    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--A3C_path", type=str, default="src/trained_models")
    parser.add_argument("--output_path", type=str, default="src/output")
    parser.add_argument("--CAE_path", type=str, default="src/pretrained/trained_cae/trained_cae_weights12-gray.pth")
    parser.add_argument("--CAE_path2", type=str, default="src/trained_models/CAE_super_mario_bros_1_1_enc2")
    parser.add_argument("--a3c_only", type=bool, default=False)
    args = parser.parse_args()
    return args
