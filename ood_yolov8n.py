import time
import os

import numpy as np
import torch
from torch.autograd import Variable
from ultralytics import YOLO
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data.build import build_yolo_dataset, build_dataloader
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import YOLODataset

from ood_utils import get_measures, arg_parser
import log

############################################################

# Code copied from https://github.com/KingJamesSong/RankFeat

############################################################

# TODO: Hay que hacer que esto carge los dataloaders de el dataset In-Distribution y Out-Distribution
def make_id_ood(args, logger):
    """Returns train and ood datasets."""
    # crop = 480

    # val_tx = tv.transforms.Compose([
    #     tv.transforms.Resize((crop, crop)),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    # in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    # out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    #in_set = 

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def create_YOLO_dataset_and_dataloader(dataset_yaml_file_name_or_path, batch_size: int):

    # TODO: En overrides se definirian ciertos parametros que se quieran tocar de la configuracion por defecto,
    # de tal forma que get_cfg() se encarga de coger esa configuracion por defecto y sobreescribirla con 
    # lo que nosotros hayamos definido en overrides. Lo mejor para definir estos overrides es sacarlos tanto de
    # otro archivo de configuracion que nosotros cargemos como queramos (desde un YAML o de un script de python)
    # como sacarlo del argparser
    overrides = dict()
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)

    data_dict = check_det_dataset(dataset_yaml_file_name_or_path)
    imgs_path = data_dict['val']

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=imgs_path,  # Path to the folder containing the images
        batch=batch_size,
        data=data_dict,  # El data dictionary que se puede sacar de data = check_det_dataset(self.args.data)
        mode='test',
        rect=False,
        stride=32
    )

    dataloader = build_dataloader(
        dataset=dataset,
        batch=dataset.batch_size,
        workers=cfg.workers,
        shuffle=False,
        rank=-1  # For distributed computing, leave -1 if no distributed computing is done
    )

    return dataset, dataloader


# Maximum Logit Score
def iterate_data_msp(data_loader, model: YOLO, device):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    all_scores = []
    print('AAAAAAAAAAAAAAAAAAAAAAA')
    for idx, data in enumerate(data_loader):
        print('BBBBBBBBBBBBBBBBBB')
        # Esto es para sacar imagenes y labels en diferentes variables
        if isinstance(data, dict):
            imgs = data['img'].to(device)
        else:
            imgs, targets = data

        # Procesar imagenes en el modelo para obtener logits y las cajas
        result = model.predict(imgs, save=True)
        # print(result)

        # Matchear cuales de las cajas han sido predichas correctamente


        # Formar un array de la forma [nº de caja, clase y score]
        score_and_cls_per_bbox = ()

        # Acumular en una lista los scores de cada caja
        # all_scores.append()

        # conf, _ = torch.max(m(logits), dim=-1)
        # confs.extend(conf.data.cpu().numpy())

    # return np.array(confs)
    return None
# ODIN Score
def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)


        outputs =  model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

    return np.array(confs)

# Energy Score
def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def run_eval(model, device, in_loader, out_loader, logger, args, num_classes):
    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model, device)
        quit()
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model, device)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
        """
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes)
    elif args.score == 'RankFeat':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_rankfeat(in_loader, model, args.temperature_rankfeat)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_rankfeat(out_loader, model, args.temperature_rankfeat)
    elif args.score == 'React':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_react)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_react)
        """
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()


def main(args):
    print('----------------------------')
    print('****************************')
    print('****************************')
    print('****************************')

    logger = log.setup_logger(args)

    # TODO: This is for reproducibility 
    # torch.backends.cudnn.benchmark = True

    # TODO: Unused till we implement GradNorm
    if args.score == 'GradNorm':
        args.batch = 1

    # Load ID data and OOD data
    # ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader('coco128_custom.yaml', batch_size=args.batch_size)
    ind_dataset, ind_dataloader = create_YOLO_dataset_and_dataloader('coco128_custom.yaml', batch_size=5)
    # ood_dataset, ood_dataloader = create_YOLO_dataset_and_dataloader('coco128_custom.yaml', batch_size=args.batch_size)
    ood_dataloader = None
    # in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)
    print(ind_dataset)

    # TODO: usar el argparser para elegir el modelo que queremos cargar
    model_to_load = 'yolov8n.pt'

    logger.info(f"Loading model {model_to_load} in {args.device}")

    # Load YOLO model
    # TODO: add different YOLO models
    model = YOLO(model_to_load) 
    # state_dict = torch.load(args.model_path)
    # model.load_state_dict_custom(state_dict['model'])

    # TODO: Unused till we implement GradNorm
    # if args.score != 'GradNorm':
    #     model = torch.nn.DataParallel(model)

    # Extract the number of classes
    if isinstance(ind_dataset, YOLODataset):
        num_classes = len(ind_dataset.data['names'])
    else:
        num_classes = len(ind_dataset.classes)

    start_time = time.time()
    run_eval(model, args.device, ind_dataloader, ood_dataloader, logger, args, num_classes=num_classes)
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()
    
    parser.add_argument('-d', '--device', default='cuda', type=str, help='use cpu or cuda')
    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React'], default='MSP')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')
    """
    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=float,
                        help='temperature scaling for GradNorm')
    # arguments for React
    parser.add_argument('--temperature_react', default=1, type=float,
                        help='temperature scaling for React')
    # arguments for RankFeat
    parser.add_argument('--temperature_rankfeat', default=1, type=float,
                        help='temperature scaling for RankFeat')
    """
    print('******************************************')
    main(parser.parse_args())