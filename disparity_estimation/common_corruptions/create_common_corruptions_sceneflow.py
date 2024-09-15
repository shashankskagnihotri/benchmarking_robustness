import os 
from imagecorruptions import get_corruption_names, corrupt
from PIL import Image
import numpy as np
import concurrent.futures
from dataloader import get_dataset
import logging


# Config of Loggings
logging.basicConfig(
    level=logging.INFO,  # oder ein anderes gewünschtes Log-Level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logfile_sceneflow.log'),  # Dateihandler für Log-Datei
        logging.StreamHandler()  # Stream-Handler für Konsole
    ]
)

# Create Logger
logger = logging.getLogger()

# Beispiel-Log-Nachrichten
logger.info('Begin of corruption')

import sys
start_at = int(sys.argv[1])


# FlyingThings3D
def process_batch(batch_indices, dataloader, corruption_names):
    for i in batch_indices:
        image_left_path = dataloader.img_left_filenames[i]
        image_right_path = dataloader.img_right_filenames[i]
        image_left = dataloader.load_image(image_left_path)
        image_right = dataloader.load_image(image_right_path)

        for corruption in corruption_names:
            for severity in range(1, 6):
                
                image_left_path_corrupted = image_left_path.replace('FlyingThings3D', f'FlyingThings3D/Common_corruptions/{corruption}/severity_{severity}')
                image_right_path_corrupted = image_right_path.replace('FlyingThings3D', f'FlyingThings3D/Common_corruptions/{corruption}/severity_{severity}')
                
                if os.path.isfile(image_left_path_corrupted) and os.path.isfile(image_right_path_corrupted):
                    logger.info(f'{i} {len(dataloader)} images corrupted {corruption} {severity} alredy exists')
                    continue
                else:
                    print(f'{i} {len(dataloader)} images corrupted {corruption} {severity} not exists')
                
                image_left_arr = np.array(image_left)
                image_right_arr = np.array(image_right)

                corrupted_left = corrupt(image_left_arr, corruption_name=corruption, severity=severity)
                corrupted_right = corrupt(image_right_arr, corruption_name=corruption, severity=severity)

                os.makedirs(os.path.dirname(image_left_path_corrupted), exist_ok=True)
                os.makedirs(os.path.dirname(image_right_path_corrupted), exist_ok=True)

                Image.fromarray(corrupted_left).save(image_left_path_corrupted)
                Image.fromarray(corrupted_right).save(image_right_path_corrupted)

                # Beispiel-Log-Nachrichten
                logger.info(f'{i} {len(dataloader)} images corrupted {corruption} {severity}')
                # print(f'{i} {len(dataloader)} images corrupted {corruption} {severity}')


def parallel_process(dataloader):
    corruption_names = get_corruption_names()
    num_images = len(dataloader)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_batch, [i], dataloader, corruption_names) for i in range(0, 4370)]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Rufe result auf, um mögliche Ausnahmen zu erfassen

# get dataset
dataloader = get_dataset('sceneflow','/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D','test','')

print(f'len dataloader: {len(dataloader)}')

# parallel_process(dataloader)

for i in range(start_at, len(dataloader)):
    process_batch([i], dataloader, get_corruption_names())

# created = 0
# not_created = 0

# corruption_names = get_corruption_names()

# for i in range(len(dataloader)):
#     image_left_path = dataloader.img_left_filenames[i]
#     image_right_path = dataloader.img_right_filenames[i]
#     image_left = dataloader.load_image(image_left_path)
#     image_right = dataloader.load_image(image_right_path)

#     for corruption in corruption_names:
#         for severity in range(5):
            
#             image_left_path_corrupted = image_left_path.replace('FlyingThings3D', f'FlyingThings3D/Common_corruptions/{corruption}/severity_{severity}')
#             image_right_path_corrupted = image_right_path.replace('FlyingThings3D', f'FlyingThings3D/Common_corruptions/{corruption}/severity_{severity}')
            
#             if os.path.isfile(image_left_path_corrupted) and os.path.isfile(image_right_path_corrupted):
#                 # logger.info(f'{i} {len(dataloader)} images corrupted {corruption} {severity} alredy exists')
#                 created += 1
#             else:
#                 not_created += 1

#     print(f'{i} created: {created}, not created: {not_created}')

# print(f'created: {created}, not created: {not_created}')


logging.shutdown()