import codecs
from aip import AipSpeech
from pydub import AudioSegment
import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

from tkinter import *
from tkVideoPlayer import TkinterVideo

for i in range(10000):

    text = input("请输入文本:")
    with codecs.open("result/voice_input.txt", "w", "utf-8-sig") as file:
        file.write(text)

       # 打开第一个 txt 文件，并读取其中的内容
    with open("result/voice_input.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # 判断内容是否包含"你好啊"
    if content.find("你好啊") > -1:
        # 输出内容到第二个 txt 文件中
        with open("result/answer.txt", "w", encoding="utf-8") as f:
            f.write("我是卖货机器人，欢迎选购我们的商品,我希望我今天可以卖掉所有的东西\n")
    else:
        with open("result/answer.txt", "w", encoding="utf-8") as f:
            f.write("请说你好啊，不然我不知道说什么\n")


    APP_ID = '33857383'
    API_KEY = 'PRYLs6vn2ZedNlYSeZp8nhT0'
    SECRET_KEY = 'WbnYIeFv7lM4TO9NC32BEK1GhBjAzokI'

    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    with open("result/answer.txt", "r", encoding="utf-8") as file:
        content = file.read()

    result = client.synthesis(content, 'zh', 1, {'vol': 5, 'per': 4})

    if not isinstance(result, dict):
        with open('input/mp3/test.mp3', 'wb') as f:
            f.write(result)

    # Step 2, convert the mp3 file to wav file
    sound = AudioSegment.from_mp3('input/mp3/test.mp3')
    sound.export("./input/audio/answer.wav", format="wav")

    # src.generate_facerender_batch 包中的 get_facerender_data 函数，用于从数据集中获取动画数据。
    # src.generate_batch 包中的 get_data 函数，用于从数据集中获取数据。
    # src.facerender.animate 包中的 AnimateFromCoeff 函数，用于将频谱系数转换为动画。
    # src.test_audio2coeff 包中的 Audio2Coeff 函数，用于将音频数据转换为频谱系数。




    def main(args):
        # torch.backends.cudnn.enabled = False

        # 获取命令行参数中的参数的值，并将其存储到一个变量中。

        pic_path = args.source_image
        audio_path = args.driven_audio
        save_dir = os.path.join(args.result_dir)
        os.makedirs(save_dir, exist_ok=True)
        pose_style = args.pose_style
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll
        ref_eyeblink = args.ref_eyeblink
        ref_pose = args.ref_pose

        ##获取命令行参数中的第一个路径，作为当前代码的路径。
        current_code_path = sys.argv[0]
        # 将当前代码的路径转换为绝对路径，并获取根目录的路径，作为当前项目的根目录。
        current_root_path = os.path.split(current_code_path)[0]
        # 将当前项目的根目录设置为 torch 库的备份路径，以便在训练过程中备份模型和配置文件。
        os.environ['TORCH_HOME'] = os.path.join(current_root_path, args.checkpoint_dir)
        # 将当前项目的根目录设置为模型备份路径，存储模型中的人脸关键点预测器。
        path_of_lm_croper = os.path.join(current_root_path, args.checkpoint_dir, 'shape_predictor_68_face_landmarks.dat')
        # 将当前项目的根目录设置为模型备份路径，存储网络重建模型。
        path_of_net_recon_model = os.path.join(current_root_path, args.checkpoint_dir, 'epoch_20.pth')
        # 将当前项目的根目录设置为模型备份路径，存储用于姿态估计的BFM模型。
        dir_of_BFM_fitting = os.path.join(current_root_path, args.checkpoint_dir, 'BFM_Fitting')
        # 将当前项目的根目录设置为模型备份路径，存储音频到唇语转换模型。(原为 wav2lip.pth )
        wav2lip_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'wav2lip.pth')
        # 将当前项目的根目录设置为模型备份路径，存储音频到姿态估计模型。
        audio2pose_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2pose_00140-model.pth')
        audio2pose_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2pose.yaml')
        # 将当前项目的根目录设置为模型备份路径，存储音频到面部表情估计模型。
        audio2exp_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'auido2exp_00300-model.pth')
        audio2exp_yaml_path = os.path.join(current_root_path, 'src', 'config', 'auido2exp.yaml')
        # 将当前项目的根目录设置为模型备份路径，存储面部表情估计模型。
        free_view_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'facevid2vid_00189-model.pth.tar')
        # 将当前项目的根目录设置为配置文件备份路径，存储模型和配置文件(.pth.tar是训练好的模型)
        # 将当前项目的根目录设置为配置文件备份路径，存储面部渲染模型的 YAML 配置文件。
        if args.preprocess == 'full':
            mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00109-model.pth.tar')
            facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender_still.yaml')
        else:
            mapping_checkpoint = os.path.join(current_root_path, args.checkpoint_dir, 'mapping_00229-model.pth.tar')
            facerender_yaml_path = os.path.join(current_root_path, 'src', 'config', 'facerender.yaml')

        # init model
        # 主要用来对 BFM(Body Force Modeling) 模型进行预处理，以便在训练或推理时进行使用。
        print(path_of_net_recon_model)

        # 调用CropAndExtract函数，该函数接受三个参数：路径_of_lm_croper、path_of_net_recon_model 和 dir_of_BFM_fitting。
        # path_of_lm_croper 指定了用于裁剪身体模型的本地文件路径。
        # path_of_net_recon_model 是 BFM 模型的云端地址或本地地址。
        # dir_of_BFM_fitting 是用于 BFM 模型校准的训练数据文件夹。
        # CropAndExtract 函数将返回一个 preprocess_model 变量，该变量指定了 BFM 模型的预处理模型。
        preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

        print(audio2pose_checkpoint)
        print(audio2exp_checkpoint)

        # 调用 Audio2Coeff 函数，该函数接受三个参数:audio2pose_checkpoint、audio2pose_yaml_path、audio2exp_checkpoint 和 audio2exp_yaml_path。
        # audio2pose_checkpoint 和 audio2exp_checkpoint 是 BFM 模型训练过程中的 Checkpoint 文件路径。
        # audio2pose_yaml_path 和 audio2exp_yaml_path 是用于保存音频和运动数据的 YAML 文件路径。
        # Audio2Coeff 函数将返回一个 audio_to_coeff 变量，该变量指定了 BFM 模型中使用的音频到系数的变换器。
        audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path,
                                     audio2exp_checkpoint, audio2exp_yaml_path,
                                     wav2lip_checkpoint, device)

        print(free_view_checkpoint)
        print(mapping_checkpoint)

        # 调用 AnimateFromCoeff 函数，该函数接受三个参数:free_view_checkpoint、mapping_checkpoint 和 facerender_yaml_path。
        # free_view_checkpoint 和 mapping_checkpoint 是 BFM 模型训练过程中的 Checkpoint 文件路径。
        # facerender_yaml_path 是用于保存面部渲染数据的 YAML 文件路径。
        # AnimateFromCoeff 函数将返回一个 animate_from_coeff 变量，该变量指定了 BFM 模型中使用的面部渲染器。
        animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint,
                                              facerender_yaml_path, device)

        # crop image and extract 3dmm from image（裁剪图像并提取 3DMM）
        # 将输入的图像 (pic_path) 保存到 save_dir 文件夹中，并将 first_frame_dir 文件夹创建出来 (exist_ok=True)。
        # 使用 preprocess_model.generate() 函数生成 3DMM 的系数 (first_coeff_path)、裁剪后的图像 (crop_pic_path) 和裁剪信息 (crop_info)。
        # 如果 first_coeff_path 为 None，则表示无法获取输入图像的 3DMM 系数，则输出错误信息并退出。
        # 如果 ref_eyeblink 不为 None，则将 ref_eyeblink 保存到 ref_eyeblink_frame_dir 文件夹中，并将 ref_eyeblink_videoname、ref_eyeblink_frame_dir、ref_pose_videoname、ref_pose_frame_dir 保存到 args.preprocess 中。
        # 如果 ref_pose 不为 None，则将 ref_pose 保存到 ref_pose_frame_dir 文件夹中，并将 ref_pose_videoname、ref_pose_frame_dir、ref_eyeblink_videoname、ref_eyeblink_frame_dir 保存到 args.preprocess 中。
        # 分别获取 ref_eyeblink 和 ref_pose 的 3DMM 系数，并将其保存到相应的 3DMM 系数文件夹中。
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir, args.preprocess)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir)
        else:
            ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir)
        else:
            ref_pose_coeff_path = None

        # audio2ceoff（将音频转换为视频）
        # 使用 get_data() 函数获取输入音频文件和 3DMM 系数文件 (first_coeff_path 和 ref_eyeblink_coeff_path) 以及输入图像文件 (audio_path)。
        # 其中，get_data() 函数的具体实现未知，需要根据实际情况编写。
        # 使用 audio_to_coeff.generate() 函数生成音频到系数的变换矩阵 (coeff_path)。
        # 该函数需要传入 batch、save_dir、pose_style、ref_pose_coeff_path 参数。
        # batch 是一个包含音频和图像的列表，每个元素都是一个元组，包含音频文件的路径和图像文件的路径。
        # save_dir 是保存系数的文件夹路径，pose_style 是用于控制面部姿态的系数
        # ref_pose_coeff_path 是用于参考的面部姿态系数文件路径。
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

        # 3dface render（在视频中加入 3D 面部渲染效果）
        # 如果 args.face3dvis 为真，则使用 gen_composed_video() 函数生成视频。
        # 该函数需要传入 args、device、first_coeff_path、coeff_path、audio_path、save_dir 六个参数。
        # 其中，args 是一个包含参数的字典，device 是用于渲染的 GPU 设备路径，first_coeff_path 和 coeff_path 是用于转换音频的系数文件路径，
        # audio_path 是输入音频文件的路径，save_dir 是保存视频的文件夹路径。
        # 生成视频后，将视频保存到 save_dir 文件夹中，文件名为 3dface.mp4。
        if args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

        # coeff2video（将系数转换为视频，并在视频中加入面部动画效果）
        # 使用 get_facerender_data() 函数获取输入的系数文件 (coeff_path)、裁剪后的图像文件 (crop_pic_path)、输入的图像文件 (pic_path)、以及输入的音频文件 (audio_path)。
        # 其中，batch_size、input_yaw_list、input_pitch_list、input_roll_list 等参数需要根据实际情况进行修改。
        # 同时，该函数的具体实现未知，需要根据实际情况编写。
        # 将获取到的面部动画数据 (data) 传递给 animate_from_coeff.generate() 函数。
        # 该函数需要传入 save_dir、pic_path、crop_info、enhancer、background_enhancer、preprocess 参数。
        # 其中，save_dir 是保存视频的文件夹路径，pic_path 是输入的图像文件路径，crop_info 是用于裁剪图像的信息，
        # enhancer 和 background_enhancer 是用于增强面部动画效果的功能参数，preprocess 是用于预处理图像的参数。
        # 使用 animate_from_coeff.generate() 函数生成视频。
        # 该函数会将面部动画效果添加到输入的图像上，并生成一个新的视频文件。生成的视频文件将保存在 save_dir 文件夹中。
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                                   batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                   expression_scale=args.expression_scale, still_mode=args.still,
                                   preprocess=args.preprocess)

        animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                    enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                    preprocess=args.preprocess)


    if __name__ == '__main__':

        # 首先定义了一个名为 ArgumentParser 的函数，用于解析命令行参数。该函数返回一个 ArgumentParser 对象。
        parser = ArgumentParser()
        # 图片及音频路径
        parser.add_argument("--driven_audio", default='./input/audio/answer.wav', help="path to driven audio")
        parser.add_argument("--source_image", default='./result/background_image/background_image1.png',
                            help="path to source image")
        # ref_eyeblink 和 ref_pose，用于指定输入的参考视频，提供眼睛闪烁和姿态信息。
        parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
        parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
        # 解析输入和输出的 checkpoint_dir 和 result_dir
        parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
        parser.add_argument("--result_dir", default='./result/video/answer.mp4', help="path to output")
        # pose_style、batch_size、expression_scale 等，这些参数用于控制面部动画效果。
        parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
        parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
        parser.add_argument("--expression_scale", type=float, default=1., help="the batch size of facerender")
        # input_yaw、input_pitch、input_roll 等，用于指定用户输入的旋转角度
        parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
        parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
        parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
        # enhancer 和 background_enhancer，用于指定面部增强和背景增强的算法
        parser.add_argument('--enhancer', type=str, default='gfpgan', help="Face enhancer, [gfpgan, RestoreFormer]")
        parser.add_argument('--background_enhancer', type=str, default=None, help="background enhancer, [realesrgan]")
        # 设置了 CPU 选项，是否生成 3D 面部， 3D 关键点，是否进行全身动画效果的处理。
        parser.add_argument("--cpu", dest="cpu", action="store_true")
        parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks")
        parser.add_argument("--still", action="store_true",
                            help="can crop back to the original videos for the full body aniamtion")
        parser.add_argument("--preprocess", default='crop', choices=['crop', 'resize', 'full'],
                            help="how to preprocess the images")

        # net structure and parameters
        # 用于指定神经网络重建器的类型。
        parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'],
                            help='useless')
        # 指定初始化网络的参数文件的路径。
        parser.add_argument('--init_path', type=str, default=None, help='Useless')
        # 指定是否初始化最后一个全连接层的权重。
        parser.add_argument('--use_last_fc', default=False, help='zero initialize the last fc')
        # 指定用于加载 BFM 模型的文件夹的路径。
        parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
        # 用于指定 BFM 模型的文件名。
        parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

        # default renderer parameters
        # 定义了一些默认的渲染参数，包括焦距、中心点、相机深度、近距点和远距点。这些参数的类型均为浮点数，默认值比较合适。
        # 初始数值1015.   112.   10.   5.   15.
        parser.add_argument('--focal', type=float, default=1015.)
        parser.add_argument('--center', type=float, default=112.)
        parser.add_argument('--camera_d', type=float, default=10.)
        parser.add_argument('--z_near', type=float, default=5.)
        parser.add_argument('--z_far', type=float, default=15.)
        # 解析命令行参数。
        args = parser.parse_args()
        # 根据 torch.cuda.is_available() 函数的结果来决定使用 CPU 还是 CUDA 加速计算。
        # 如果 CUDA 可用且 args.cpu 为 False，则将 args.device 设置为 "cuda",否则为 "cpu"。

        if torch.cuda.is_available() and not args.cpu:
            args.device = "cuda"
        else:
            args.device = "cpu"
        # 执行应用程序的主循环
        main(args)
        open("./result/video/background_image1##answer_enhanced.mp4")








