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
    filename='logfile.log',  # Dateiname für die Log-Datei
    filemode='a'  # 'a' für Anfügen, 'w' für Überschreiben
)

# Create Logger
logger = logging.getLogger()

# Beispiel-Log-Nachrichten
logger.info('Begin of corruption')



# FlyingThings3D
def process_batch(batch_indices, dataloader, corruption_names):
    for i in batch_indices:
        image_left_path = dataloader.img_left_filenames[i]
        image_right_path = dataloader.img_right_filenames[i]
        image_left = dataloader.load_image(image_left_path)
        image_right = dataloader.load_image(image_right_path)

        for corruption in corruption_names:
            for severity in range(5):
                image_left_arr = np.array(image_left)
                image_right_arr = np.array(image_right)

                corrupted_left = corrupt(image_left_arr, corruption_name=corruption, severity=severity+1)
                corrupted_right = corrupt(image_right_arr, corruption_name=corruption, severity=severity+1)

                image_left_path_corrupted = image_left_path.replace('FlyingThings3D', f'FlyingThings3D/Common_corruptions/{corruption}/severity_{severity}')
                image_right_path_corrupted = image_right_path.replace('FlyingThings3D', f'FlyingThings3D/Common_corruptions/{corruption}/severity_{severity}')
                os.makedirs(os.path.dirname(image_left_path_corrupted), exist_ok=True)
                os.makedirs(os.path.dirname(image_right_path_corrupted), exist_ok=True)

                Image.fromarray(corrupted_left).save(image_left_path_corrupted)
                Image.fromarray(corrupted_right).save(image_right_path_corrupted)

                # Beispiel-Log-Nachrichten
                logger.info(f'{i} {len(dataloader)} images corrupted {corruption} {severity}')
                # print(f'{i} {len(dataloader)} images corrupted {corruption} {severity}')

def split_indices(num_items, batch_size):
    return [range(i, min(i + batch_size, num_items)) for i in range(0, num_items, batch_size)]

def parallel_process(dataloader):
    corruption_names = get_corruption_names()
    num_images = len(dataloader)
    batch_size = 1
    batches = split_indices(num_images, batch_size)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch, dataloader, corruption_names) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Rufe result auf, um mögliche Ausnahmen zu erfassen

# get dataset
dataloader = get_dataset('sceneflow','/pfs/work7/workspace/scratch/ma_faroesch-team_project_fss2024/dataset/FlyingThings3D','test','')

parallel_process(dataloader)

logging.shutdown()