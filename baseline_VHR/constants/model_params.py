NUM_CLASSES = 50
batch_size = 1
#'BATCH_SIZE': batch_size,
#'LR': 0.001,]
#
#
#'PRECISION': 32,
#
#'PRECISION': 32,
#'SEED': 42,
#                                  'PROJECT': 'Heads',
##                                  'EXPERIMENT': 'heads',
#                                  'MAXEPOCHS': 500,
#
#                                  'IMG_MEAN': [0.485, 0.456, 0.406],                              'IMG_STD': [0.229, 0.224, 0.225],
#
#
#                   'IOU_THRESHOLD': 0.5
params_fasterrcnn_resnet50_fpn = {
                                  'CLASSES': NUM_CLASSES,
                                  'BACKBONE': 'fasterrcnn_resnet50_fpn',
                                  'FPN': False,
                                  'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
                                  'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
                                  'MIN_SIZE': 2752,
                                  'MAX_SIZE': 2752

                                  }
params_resnet18 = {
                   'CLASSES': NUM_CLASSES,
                   'BACKBONE': 'resnet18',
                   'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
                   'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
                   'MIN_SIZE': 1024,
                   'MAX_SIZE': 1024
                   }

params_mobilenet_v3_large = {'CLASSES': NUM_CLASSES,
                             'BACKBONE': 'mobilenet_v3_large',
                             'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
                             'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
                             'MIN_SIZE': 1024,
                             'MAX_SIZE': 1024,
                             }

params_densenet121 = {'CLASSES': NUM_CLASSES,
                      'BACKBONE': 'densenet121',
                      'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
                      'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
                      'MIN_SIZE': 1024,
                      'MAX_SIZE': 1024,
                      }

xView_model_dict = {'densenet121': params_densenet121,
                    'mobilenet_v3_large': params_mobilenet_v3_large,
                    'fasterrcnn_resnet50_fpn': params_fasterrcnn_resnet50_fpn,
                    'resnet18': params_resnet18}
