import os, sys, random, argparse, json, copy, logging, pprint, time, shutil, csv
from itertools import chain
import argparse
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))

from data.dataset import AlfredDataset, AlfredPyTorchDataset
from models.config.configs import Config
from models.model.mmt import MultiModalTransformer
from models.nn.mrcnn import MaskRCNNDetector
from models.eval.eval_task import EvalTaskMMT
from models.eval.eval_subgoals import EvalSubgoalsMMT



def check_input(model, arg):
    for feat_arg in ['disable_feat_lang_high','disable_feat_lang_navi','disable_feat_lang_mani',
                             'disable_feat_vis', 'disable_feat_action_his',
                            'enable_feat_vis_his', 'enable_feat_posture']:

        feat_arg_eval = 'eval_'+ feat_arg
        if 'lang' in feat_arg:
            feat_arg = feat_arg[:-5]
        if not hasattr(model.args, feat_arg) or getattr(model.args, feat_arg) != getattr(arg, feat_arg_eval):
            logging.warning('WARNING: dismatch input option: %s'%feat_arg_eval)
def validation_check(model, split, task_type, data_loader, epoch):
    model.eval()

    enable_mask = 'mani' in task_type
    enable_navi_aux = 'navi' in task_type and model.args.auxiliary_loss_navi

    type_correct, arg_correct, mask_correct, cnt, arg_cnt, mask_cnt, navi_cnt =0, 0, 0, 0, 0, 0, 0
    navi_correct = {'visible': 0, 'reached': 0, 'progress': 0}
    for idx, batch in enumerate(data_loader):
        # print("batch shape")
        # print(batch)
        # print(batch['path'])
        # print(batch['vision_feats'].shape)
        # print(batch['vision_cls'].shape)
        # print(batch['lang_input'].shape)
        # print(batch['action_history_input'].shape)
        # print(batch['actype_label'].shape)
        # print(batch['arg_label'].shape)
        # print(batch['seq_mask'].shape)

        type_preds, arg_preds, mask_preds, navi_preds, labels = model(batch, False)

        cnt += len(type_preds)
        type_correct += (type_preds==labels['type']).sum().item()
        # if epoch > 4:
        #     print('type errors:')
        #     labelsss = labels['type'][type_preds!=labels['type']]
        #     print([(i.item(), labelsss[idx].item()) for idx, i in enumerate(type_preds[type_preds!=labels['type']])])
        if 'high' in task_type:
            arg_correct += (arg_preds==labels['arg']).sum().item()
        elif 'low' in task_type:
            arg_idx = labels['arg'] != -1
            arg_correct += (arg_preds[arg_idx]==labels['arg'][arg_idx]).sum().item()
            arg_cnt += arg_idx.sum().item()
            # if epoch > 4:
            #     print('arg errors:')
            #     labelsss = labels['arg'][arg_preds!=labels['arg']]
            #     print([(i.item(), labelsss[idx].item()) for idx, i in enumerate(arg_preds[arg_preds!=labels['arg']])])
            if enable_mask:
                mask_idx = labels['mask'] != -1
                mask_correct += (mask_preds[mask_idx]==labels['mask'][mask_idx]).sum().item()
                mask_cnt += mask_idx.sum().item()
            if enable_navi_aux:
                navi_idx = labels['visible'] != -1
                navi_preds['visible'][navi_idx]==labels['visible'][navi_idx]
                navi_correct['visible'] += (navi_preds['visible'][navi_idx]==labels['visible'][navi_idx]).sum().item()
                navi_correct['reached'] += (navi_preds['reached'][navi_idx]==labels['reached'][navi_idx]).sum().item()
                navi_correct['progress'] += (navi_preds['progress'][navi_idx]-labels['progress'][navi_idx]).square().sum().item()
                navi_cnt += navi_idx.sum().item()

    type_accu = type_correct/cnt
    arg_accu = arg_correct/cnt if 'high' in task_type else arg_correct/(arg_cnt + 1e-8)
    mask_accu = mask_correct/(mask_cnt+1e-8) if enable_mask else 0
    mask_str = ' |mask %.3f'%(mask_accu) if enable_mask else ''
    navi_str = ''
    if enable_navi_aux:
        visible_accu = navi_correct['visible']/(navi_cnt+1e-8)
        reached_accu = navi_correct['reached']/(navi_cnt+1e-8)
        progress_mse = navi_correct['progress']/(navi_cnt+1e-8)
        navi_str = '|vis: %.3f |rea: %.3f |prog: %.4f'%(visible_accu, reached_accu, progress_mse)

    logging.info('Validation [%12s %8s] accuracy type: %.3f |arg: %.3f%s%s'%(
        split, task_type, type_accu, arg_accu, mask_str, navi_str))

    writer.add_scalar('valid_accu/%s/%s_type'%(task_type, split), type_accu, epoch)
    writer.add_scalar('valid_accu/%s/%s_arg'%(task_type, split), arg_accu, epoch)
    if enable_mask:
        writer.add_scalar('valid_accu/%s/%s_mask'%(task_type, split), mask_accu, epoch)
    if enable_navi_aux:
        writer.add_scalar('valid_accu/%s/%s_visible'%(task_type, split), visible_accu, epoch)
        writer.add_scalar('valid_accu/%s/%s_reached'%(task_type, split), reached_accu, epoch)
        writer.add_scalar('valid_accu/%s/%s_progress'%(task_type, split), progress_mse, epoch)
    
    return type_accu, arg_accu, mask_accu

if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser()
    Config(parser)

    # eval settings
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['valid_seen', 'valid_unseen'])
    # parser.add_argument('--eval_path', type=str, default="exp/something")
    parser.add_argument('--ckpt_name', type=str, default="model_best_seen.pth")
    parser.add_argument('--num_core_per_proc', type=int, default=5, help='cpu cores used per process')
    # parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--max_high_steps', type=int, default=20, help='max steps before a high-level episode termination')
    parser.add_argument('--max_high_fails', type=int, default=5, help='max failing times to try high-level proposals')
    parser.add_argument('--max_fails', type=int, default=999, help='max failing times in ALFRED benchmark')
    parser.add_argument('--max_low_steps', type=int, default=50, help='max steps before a low-level episode termination')
    parser.add_argument('--only_eval_mask', dest='only_eval_mask', action='store_true')
    parser.add_argument('--use_gt_navigation', dest='use_gt_navigation', action='store_true')
    parser.add_argument('--use_gt_high_action', dest='use_gt_high_action', action='store_true')
    parser.add_argument('--use_gt_mask', dest='use_gt_mask', action='store_true')
    parser.add_argument('--save_video', action='store_true')

    parser.add_argument('--eval_disable_feat_lang_high', help='do not use language features as high input', action='store_true')
    parser.add_argument('--eval_disable_feat_lang_navi', help='do not use language features as low-navi input', action='store_true')
    parser.add_argument('--eval_disable_feat_lang_mani', help='do not use language features as low-mani input', action='store_true')
    parser.add_argument('--eval_disable_feat_vis', help='do not use visual features as input', action='store_true')
    parser.add_argument('--eval_disable_feat_action_his', help='do not use action history features as input', action='store_true')
    parser.add_argument('--eval_enable_feat_vis_his', help='use additional history visual features as input', action='store_true')
    parser.add_argument('--eval_enable_feat_posture', help='use additional agent posture features as input', action='store_true')


    # parse arguments
    args = parser.parse_args()
    args_model = argparse.Namespace(**json.load(open(os.path.join(args.eval_path, 'config.json'), 'r')))
    args.use_bert = args_model.use_bert
    args.bert_model = args_model.bert_model
    # args.inner_dim = 1024

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # set logger
    # log_dir = os.path.join(args.exp_path, 'train_log.log')
    # log_handlers = [logging.StreamHandler(), logging.FileHandler(log_dir)]
    # logging.basicConfig(handlers=log_handlers, level=logging.INFO,
    #     format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logging.info(pprint.pformat(args.__dict__))
    # with open(os.path.join(args.exp_path, 'config.json'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    # setup tensorboard
    # tb_log_write_dir = os.path.join('tb_log', temp, model_str)
    # if os.path.exists(tb_log_write_dir):
    #     shutil.rmtree(tb_log_write_dir)
    # writer = SummaryWriter(tb_log_write_dir)
    writer = SummaryWriter()

    # load alfred data and build pytorch data sets and loaders
    alfred_data = AlfredDataset(args)



    # setup modelbatch
    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    print(device)
    ckpt_path = os.path.join(args.eval_path, args.ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MultiModalTransformer(args_model, alfred_data)
    layer_list = model.encoder.encoder.layer
    embed_list = list(model.encoder.embeddings.parameters())
    # print(layer_list)
    #remove layers
    # remove_layers = args.remove_layers
    # if remove_layers is not "":
    #     layer_indexes = [int(x) for x in remove_layers.split(",")]
    #     layer_indexes.sort(reverse=True)
    # layer_indexes = [11,10,9,8,7,6]
    # layer_indexes.sort(reverse=True)
    # for layer_idx in layer_indexes:
        # if layer_idx < 0:
        #     print ("Only positive indices allowed")
        #     sys.exit(1)
        # del(layer_list[layer_idx])
        # print ("Removed Layer: ", layer_idx)
    model.encoder.config.num_hidden_layers = len(layer_list)
    model.load_state_dict(ckpt, strict=True)   #strict=False

    isQuantized = False
    if (isQuantized):
        torch.backends.quantized.engine = 'qnnpack'
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        torch.save(model.state_dict(), "models/quantized_bert.pth")
    # model.to(model.device)
    model.to(device)
    print(model)
    sys.exit()
    models = model

    # setup model2
    # args2 = copy.deepcopy(args)
    # # args2.eval_path = 'exp/Jan23-low-navi/ori+pos_low_navi_E-xavier256d_L4_H512_det-sep_dp0.2_di0.2_step_lr1e-04_0.999_type_sd123'
    # # args2.eval_path = 'exp/Jan23-low-navi-sep/cls+aux+pos+vishis_low_navi_E-xavier256d_L4_H512_det-sep_dp0.2_di0.2_step_lr1e-04_0.999_type_sd123'
    # args2.eval_path = 'exp/Jan23-low-navi/ori+pos+vishis_low_navi_E-xavier256d_L4_H512_det-sep_dp0.2_di0.2_step_lr1e-04_0.999_type_sd123'
    # args2.ckpt_name = 'model_best_valid.pth'
    # args2_model = argparse.Namespace(**json.load(open(os.path.join(args2.eval_path, 'config.json'), 'r')))
    # ckpt_path2 = os.path.join(args2.eval_path, args2.ckpt_name)
    # ckpt2 = torch.load(ckpt_path2, map_location=device)
    # model2 = MultiModalTransformer(args2_model, alfred_data)
    # model2.load_state_dict(ckpt2)   #
    # model2.to(model2.device)

    # models = {'navi': model2, 'mani': model}


    # log dir
    eval_type = 'task' if not args.subgoals else 'subgoal'
    gt_navi = '' if not args.use_gt_navigation else '_gtnavi'
    gt_sg = '' if not args.use_gt_high_action else '_gtsg'
    input_str = ''
    if args.eval_disable_feat_lang_high:
        input_str += 'nolanghigh_'
    if args.eval_disable_feat_lang_mani:
        input_str += 'nolangmani_'
    if args.eval_disable_feat_lang_navi:
        input_str += 'nolangnavi_'
    if args.eval_disable_feat_vis:
        input_str += 'novis_'
    if args.eval_disable_feat_action_his:
        input_str += 'noah_'
    if args.eval_enable_feat_vis_his:
        input_str += 'hasvh_'
    if args.eval_enable_feat_posture:
        input_str += 'haspos_'
    log_name = '%s_%s_%s_maxfail%d_%s%s%s.log'%(args.name_temp, eval_type, args.eval_split, args.max_high_fails,
        input_str, gt_navi, gt_sg)
    if args.debug:
        log_name = log_name.replace('.log', '_debug.log')
    if isinstance(models, dict):
        log_name = log_name.replace('.log', '_sep.log')
    args.log_dir = os.path.join(args.eval_path, log_name)

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_handlers = [logging.StreamHandler(), logging.FileHandler(args.log_dir)]
    logging.basicConfig(handlers=log_handlers, level=log_level,
        format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if isinstance(models, dict):
        logging.info('model 1: %s'%ckpt_path)
        check_input(model, args)
        logging.info('model 2: %s'%ckpt_path2)
        check_input(model2, args2)
    else:
        logging.info('model: %s'%ckpt_path)
        check_input(model, args)

    # setup object detector
    # detector = MaskRCNNDetector(args, detectors=[args.detector_type])
    '''
    # eval mode
    if args.subgoals:
        eval = EvalSubgoalsMMT(args, alfred_data, models, detector, manager)
    else:
        eval = EvalTaskMMT(args, alfred_data, models, detector, manager)


    # start threads
    # eval.run(model, detector, vocabs, task_queue, args, lock, stats, results)
    eval.start()
    '''
    
    valid_names = ['valid_seen', 'valid_unseen']
    task_types = ['high'] if args.train_level != 'low' else []
    if args.train_level != 'high' and args.low_data != 'navi':
        task_types.append('low_mani')
    if args.train_level != 'high' and args.low_data != 'mani':
        task_types.append('low_navi')
    valid_loaders = {}
    for split in valid_names:
        for tt in task_types:
            bs = args.batch
            # print("the batch size is: {}".format(args.batch))
            # sys.exit()
            # bs = 400 if args.train_proportion != 100 else args.batch
            valid_set = AlfredPyTorchDataset(alfred_data, split, tt, args)
            valid_loaders[split+'-'+tt] = torch.utils.data.DataLoader(valid_set,
                batch_size=bs, shuffle=True, num_workers=6, pin_memory=True)

    # some initilizations before loop
    best_valid, fail = {'seen':0 , 'unseen': 0}, {'seen':0 , 'unseen': 0}
    counts = {'start_time': time.time(), 'epoch': 0}
    # for lt, ld in train_loaders.items():
    #     counts['dlen_%s'%lt] = len(ld)
    #     counts['iter_%s'%lt] = 0
    # if args.use_bert and args.bert_lr_schedule:
    #     counts['scheduler'] = scheduler

    # start training loop
    args.epoch = 1 
    for epoch in range(args.epoch):
        counts['epoch'] = epoch
        # validation check
        valid_accu = {}
        # print(list(valid_loaders.items())[0])
        valid_loader_small = list(valid_loaders.items())[0]
        # sys.exit()
        with torch.no_grad():
            i=0
            for dataset_type, loader in valid_loaders.items():
                if i == 0:
                    split, tt = dataset_type.split('-')
                    type_accu, arg_accu, mask_accu = validation_check(
                        model, split, tt, loader, epoch)

                    valid_accu[dataset_type + '-' + 'type'] = type_accu
                    valid_accu[dataset_type + '-' + 'arg'] = arg_accu
                    valid_accu[dataset_type + '-' + 'mask'] =mask_accu
                i += 1

            vs_level = 'low' if args.train_level == 'mix' else args.train_level
            if vs_level == 'low':
                vs_level += '_mani' if args.low_data == 'mani' else '_navi'
            valid_scores = {
                'seen': valid_accu['valid_seen-%s-%s'%(vs_level, args.valid_metric)],
                'unseen': valid_accu['valid_unseen-%s-%s'%(vs_level, args.valid_metric)],
            }
            print(valid_accu)
            print('*'*20)
            print(valid_scores)

            valid_check_failed = False
            for valid_type, valid_score in valid_scores.items():
                logging.info("%s valid score  (%s %s): %.3f" %(valid_type, args.train_level,
                        args.valid_metric, valid_score))
                writer.add_scalar('valid_accu/valid_score_%s'%valid_type, valid_score, epoch)

                # if (valid_score - best_valid[valid_type]) >= 1e-3:
                #     best_valid[valid_type] = valid_score
                #     best_accus = valid_accu
                #     fail[valid_type] = 0
                #     save_dir = os.path.join(args.exp_path, "model_best_%s.pth"%valid_type)
                #     torch.save(model.state_dict(), save_dir)
                #     torch.save({'epoch': epoch,
                #                         'optimizer': optimizer.state_dict(),
                #                         'loss_weight': log_var,
                #                         'score': valid_score}, save_dir.replace('model_', 'state_'))
                #     logging.info("[valid %s] New best score! Model of epoch %d saved."%(valid_type, epoch))
                # else:
                #     fail[valid_type] += 1
                #     logging.info("[valid %s] score does not get better for %d epochs"%(valid_type, fail[valid_type]))
                #     valid_check_failed = True
            writer.flush()

            # if fail['seen'] >= args.early_stop and fail['unseen'] >= args.early_stop:
            #     tt = time.time() - counts['start_time']
            #     save_dir = os.path.join(args.exp_path, "model_final.pth")
            #     torch.save(model.state_dict(), save_dir)
            #     torch.save({'epoch': epoch,
            #                        'optimizer': optimizer.state_dict(),
            #                        'loss_weight': log_var,
            #                        'score': valid_scores}, save_dir.replace('model_', 'state_'))
            #     logging.info("Training early stopped. Total time: %dh%dm"%(tt//3600, tt//60%60))
            #     writer.add_hparams({k:args.__dict__[k] for k in hyperparameters},
            #           {'metric/best_valid_seen': valid_scores['seen'],
            #            'metric/best_valid_unseen': valid_scores['unseen']})

            #     write_head = not os.path.exists(temp+'_results.csv')
            #     with open(temp+'_results.csv', 'a') as rf:
            #         writer = csv.DictWriter(rf, fieldnames=['name', 'proportion', 'seed'] + list(best_accus.keys()))
            #         best_accus['name'] = model_str
            #         best_accus['proportion'] = args.train_proportion
            #         best_accus['seed'] = args.seed
            #         if write_head:
            #             writer.writeheader()
            #         writer.writerows([best_accus])
            #     quit()

    # tt = time.time() - counts['start_time']
    # logging.info("Training stopped. Total time: %dh%dm"%(tt//3600, tt//60%60))
    # save_dir = os.path.join(args.exp_path, "model_final_ep%d.pth"%(epoch))
    # torch.save(model.state_dict(), save_dir)
    # torch.save({'epoch': epoch,
    #                    'optimizer': optimizer.state_dict(),
    #                    'loss_weight': log_var,
    #                    'score': valid_scores}, save_dir.replace('model_', 'state_'))
    # writer.add_hparams({k:args.__dict__[k] for k in hyperparameters},
    #                   {'metric/best_valid_seen': valid_scores['seen'],
    #                    'metric/best_valid_unseen': valid_scores['unseen']})

    # write_head = not os.path.exists(temp+'_results.csv')
    # with open(temp+'_results.csv', 'a') as rf:
    #     writer = csv.DictWriter(rf, fieldnames=['name', 'proportion', 'seed'] + list(best_accus.keys()))
    #     best_accus['name'] = model_str
    #     best_accus['proportion'] = args.train_proportion
    #     best_accus['seed'] = args.seed
    #     if write_head:
    #         writer.writeheader()
    #     writer.writerows([best_accus])